// -*- C++ -*-
//
// Package:    RecoEgamma/PhotonIdentification
// Class:      PhotonTransformerIDProducer
//
/**\class PhotonTransformerIDProducer

 Description: transformer-based photon identification on MiniAOD photons.

 Each photon is described by three sets of tokens and 18 standard photon variables:
   - ECAL crystals: 32x32 window around the supercluster seed (EB: ieta x iphi; EE: ix x iy, with a window of the
     same angular size as in EB), as standardised log-energy image. Inside the model, up to 64 crystals within
     Delta R < 0.3 of the energy-weighted centroid are kept.
   - Packed PF candidates within Delta R < 0.3 of the photon, without the candidates associated with the photon.
   - Preshower strips (endcap only): strips within +-20 cm of the line from the primary vertex to the supercluster
     seed, at the z of the strip, plus the strips of the supercluster's preshower clusters. Inside the model, up to
     64 strips within Delta R < 0.3 are kept.
   - Shower shapes, isolations, supercluster eta and rho (standardised).
 The TorchScript model returns a logit and four learned features. The producer stores the sigmoid of the logit
 ("score", signal = 1) and the features ("feature0", ...) as edm::ValueMap<float> keyed by the input photons.

 The standardisation constants, the PF cone and the preshower window are read from a JSON file exported together
 with the model. The input conventions are those of the training (MiniAOD collections), so the producer must run
 on MiniAOD. All photons of an event are evaluated in one batch. PyTorchService must be in the process.
*/

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <memory>
#include <numbers>
#include <string>
#include <unordered_set>
#include <vector>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "PhysicsTools/PyTorch/interface/Model.h"

#include <nlohmann/json.hpp>

namespace photonTransformerID {

  constexpr int kGrid = 32;  // crystal window: kGrid x kGrid
  constexpr int kNCells = kGrid * kGrid;
  constexpr int kNHLF = 18;                 // standard photon variables
  constexpr int kPFRawDim = 13;             // PF features before the one-hot encoding of the particle type
  constexpr int kPFDim = 18;                // PF features given to the model
  constexpr int kESDim = 6;                 // preshower strip features
  constexpr float kESOffsetScaleCm = 20.f;  // dx, dy of the strips are given in units of 20 cm

  // Preprocessing constants exported together with the model.
  struct Preprocessing {
    float rechitLogEps;
    std::vector<float> cellMean, cellStd;  // kNCells each, row-major
    std::vector<float> hlfMean, hlfStd;    // kNHLF each
    float pfLogEps;
    float pfDrCone;
    std::vector<int> pfStandardizeIdx;  // indices of the raw PF features that are standardised
    std::vector<float> pfMean, pfStd;   // one per standardised index
    float esWindowHalfWidthCm;
    float esLogScaleGeV;
    float esEnergyMean, esEnergyStd;
    int nFeatures;

    static Preprocessing fromJSON(const std::string& path) {
      std::ifstream file(path);
      if (!file.is_open())
        throw cms::Exception("PhotonTransformerID") << "cannot open the preprocessing file " << path;
      Preprocessing p;
      try {
        const auto json = nlohmann::json::parse(file);
        const auto& rechit = json.at("rechit");
        p.rechitLogEps = rechit.at("log_eps").get<float>();
        for (const auto& row : rechit.at("cell_mean"))
          for (const auto& value : row)
            p.cellMean.push_back(value.get<float>());
        for (const auto& row : rechit.at("cell_std"))
          for (const auto& value : row)
            p.cellStd.push_back(value.get<float>());
        p.hlfMean = json.at("hlf").at("feat_mean").get<std::vector<float>>();
        p.hlfStd = json.at("hlf").at("feat_std").get<std::vector<float>>();
        const auto& pf = json.at("pf");
        p.pfLogEps = pf.at("log_eps").get<float>();
        p.pfDrCone = pf.at("dr_cone").get<float>();
        p.pfStandardizeIdx = pf.at("standardize_idx").get<std::vector<int>>();
        p.pfMean = pf.at("x_mean").get<std::vector<float>>();
        p.pfStd = pf.at("x_std").get<std::vector<float>>();
        const auto& es = json.at("es");
        p.esWindowHalfWidthCm = es.at("window_half_width_cm").get<float>();
        p.esLogScaleGeV = es.at("energy_log_scale_gev").get<float>();
        p.esEnergyMean = es.at("energy_mean").get<float>();
        p.esEnergyStd = es.at("energy_std").get<float>();
        p.nFeatures = json.at("model").at("n_features").get<int>();
      } catch (const nlohmann::json::exception& e) {
        throw cms::Exception("PhotonTransformerID") << "invalid preprocessing file " << path << ": " << e.what();
      }
      const bool pfIndicesValid = std::all_of(p.pfStandardizeIdx.begin(), p.pfStandardizeIdx.end(), [](int i) {
        return i >= 0 && i < kPFRawDim && i != 4 && i != 5;
      });
      if (p.cellMean.size() != kNCells || p.cellStd.size() != kNCells || p.hlfMean.size() != kNHLF ||
          p.hlfStd.size() != kNHLF || p.pfMean.size() != p.pfStandardizeIdx.size() ||
          p.pfStd.size() != p.pfStandardizeIdx.size() || !pfIndicesValid || p.nFeatures < 1)
        throw cms::Exception("PhotonTransformerID") << "inconsistent preprocessing constants in " << path;
      return p;
    }
  };

  // Model and preprocessing, shared by all streams.
  struct Cache {
    explicit Cache(const edm::ParameterSet& cfg)
        : model(std::make_unique<cms::torch::Model>(cfg.getParameter<edm::FileInPath>("modelFile").fullPath())),
          preprocessing(Preprocessing::fromJSON(cfg.getParameter<edm::FileInPath>("preprocessingFile").fullPath())) {}
    // Inference with a TorchScript module in eval mode (no gradients) does not modify it and can run concurrently.
    const std::unique_ptr<cms::torch::Model> model;
    const Preprocessing preprocessing;
  };

}  // namespace photonTransformerID

namespace {

  using namespace photonTransformerID;

  // (x + pi) mod 2pi - pi, as in the training preprocessing (numpy)
  float wrapPhi(float dphi) {
    constexpr float pi = std::numbers::pi_v<float>;
    float r = std::fmod(dphi + pi, 2.f * pi);
    if (r < 0.f)
      r += 2.f * pi;
    return r - pi;
  }

  int pfTypeIndex(int pdgId) {
    switch (std::abs(pdgId)) {
      case 211:
        return 1;
      case 11:
        return 2;
      case 13:
        return 3;
      case 22:
        return 4;
      case 130:
        return 5;
      default:
        return 0;
    }
  }

  // Window of EE crystals with the angular size of kGrid EB crystals.
  struct EEWindowShape {
    int nx;
    int ny;
  };

  double wrappedDeltaPhi(double phi1, double phi2) { return std::atan2(std::sin(phi1 - phi2), std::cos(phi1 - phi2)); }

  EEWindowShape eeWindowShape(double seedEta, double seedPhi) {
    constexpr double ebCrystalAngularSize = 0.0174;
    constexpr double eeCrystalFrontSizeCm = 2.862;
    constexpr double eeFrontFaceZCm = 315.4;
    if (!std::isfinite(seedEta) || !std::isfinite(seedPhi) || seedEta == 0.0)
      return {kGrid, kGrid};
    const double z = std::copysign(eeFrontFaceZCm, seedEta);
    const double radius = z / std::sinh(seedEta);
    const double seedX = radius * std::cos(seedPhi);
    const double seedY = radius * std::sin(seedPhi);
    const auto angularDistance = [&](double x, double y) {
      const double eta = std::asinh(z / std::hypot(x, y));
      const double phi = std::atan2(y, x);
      return std::hypot(eta - seedEta, wrappedDeltaPhi(phi, seedPhi));
    };
    const double pitchX = 0.5 * (angularDistance(seedX + eeCrystalFrontSizeCm, seedY) +
                                 angularDistance(seedX - eeCrystalFrontSizeCm, seedY));
    const double pitchY = 0.5 * (angularDistance(seedX, seedY + eeCrystalFrontSizeCm) +
                                 angularDistance(seedX, seedY - eeCrystalFrontSizeCm));
    const double targetWidth = kGrid * ebCrystalAngularSize;
    const auto numberOfCrystals = [&](double pitch) {
      if (!std::isfinite(pitch) || pitch <= 0.0)
        return kGrid;
      const double n = std::floor(targetWidth / pitch + 0.5);
      return static_cast<int>(std::max(1.0, std::min(static_cast<double>(kGrid), n)));
    };
    return {numberOfCrystals(pitchX), numberOfCrystals(pitchY)};
  }

  // Standardised kGrid x kGrid log-energy image of the ECAL crystals around the supercluster seed.
  // EB: ieta x iphi window of 33 x 33 crystals centred on the seed, trimmed to 32 x 32 by dropping the outer row
  // (column) with the smaller energy sum, as in training (no iphi wrap-around).
  // EE: ix x iy window with the angular size of kGrid EB crystals, with the seed at cell (kGrid / 2, kGrid / 2).
  void fillImage(const pat::Photon& photon,
                 const EcalRecHitCollection& ebHits,
                 const EcalRecHitCollection& eeHits,
                 const Preprocessing& prep,
                 float* image) {
    const auto standardise = [&](const std::vector<float>& energies) {
      for (int cell = 0; cell < kNCells; ++cell) {
        float value = std::log(std::max(0.f, energies[cell]) + prep.rechitLogEps);
        const float sigma = prep.cellStd[cell];
        value = sigma > 0.f ? (value - prep.cellMean[cell]) / sigma : 0.f;
        image[cell] = std::isfinite(value) ? value : 0.f;
      }
    };
    std::vector<float> energies(kNCells, 0.f);
    const DetId seedId = photon.superCluster()->seed()->seed();

    if (seedId.subdetId() == EcalEndcap) {
      const EEDetId seed(seedId);
      const reco::CaloCluster& seedCluster = *photon.superCluster()->seed();
      const EEWindowShape window = eeWindowShape(seedCluster.eta(), seedCluster.phi());
      const int lowerX = -(window.nx / 2), lowerY = -(window.ny / 2);
      const int upperX = lowerX + window.nx - 1, upperY = lowerY + window.ny - 1;
      for (const auto& hit : eeHits) {
        const EEDetId id(hit.detid());
        if (id.zside() != seed.zside())
          continue;
        const int dix = id.ix() - seed.ix();
        const int diy = id.iy() - seed.iy();
        if (dix < lowerX || dix > upperX || diy < lowerY || diy > upperY)
          continue;
        energies[(kGrid / 2 + dix) * kGrid + (kGrid / 2 + diy)] = hit.energy();
      }
      standardise(energies);
      return;
    }

    constexpr int half = kGrid / 2;
    constexpr int raw = 2 * half + 1;
    std::vector<float> window(raw * raw, 0.f);
    if (seedId.subdetId() == EcalBarrel) {
      const EBDetId seed(seedId);
      for (const auto& hit : ebHits) {
        const EBDetId id(hit.detid());
        const int dieta = id.ieta() - seed.ieta();
        const int diphi = id.iphi() - seed.iphi();
        if (std::abs(dieta) > half || std::abs(diphi) > half)
          continue;
        window[(dieta + half) * raw + (diphi + half)] = hit.energy();
      }
    }
    float firstRow = 0.f, lastRow = 0.f, firstCol = 0.f, lastCol = 0.f;
    for (int i = 0; i < raw; ++i) {
      firstRow += window[i];
      lastRow += window[(raw - 1) * raw + i];
      firstCol += window[i * raw];
      lastCol += window[i * raw + raw - 1];
    }
    const int startRow = firstRow <= lastRow ? 1 : 0;
    const int startCol = firstCol <= lastCol ? 1 : 0;
    for (int r = 0; r < kGrid; ++r)
      for (int c = 0; c < kGrid; ++c)
        energies[r * kGrid + c] = window[(startRow + r) * raw + (startCol + c)];
    standardise(energies);
  }

  // Standardised shower shapes, isolations, supercluster eta and rho.
  void fillHLF(const pat::Photon& photon, float rho, const Preprocessing& prep, float* hlf) {
    const reco::SuperCluster& sc = *photon.superCluster();
    const auto& shower = photon.full5x5_showerShapeVariables();
    hlf[0] = sc.rawEnergy();
    hlf[1] = photon.full5x5_r9();
    hlf[2] = shower.sigmaIetaIeta;
    hlf[3] = sc.etaWidth();
    hlf[4] = sc.phiWidth();
    hlf[5] = shower.sigmaIetaIphi;
    hlf[6] = shower.e2x2 / (shower.e5x5 + 1e-9f);
    hlf[7] = photon.hadronicOverEm();
    hlf[8] = photon.ecalPFClusterIso();
    hlf[9] = photon.hcalPFClusterIso();
    hlf[10] = photon.trkSumPtHollowConeDR03();
    hlf[11] = photon.trkSumPtSolidConeDR04();
    hlf[12] = photon.chargedHadronIso();
    hlf[13] = photon.chargedHadronWorstVtxIso();
    hlf[14] = sc.eta();
    hlf[15] = rho;
    hlf[16] = shower.effSigmaRR;
    hlf[17] = sc.preshowerEnergy() / sc.rawEnergy();
    for (int k = 0; k < kNHLF; ++k) {
      const float value = (hlf[k] - prep.hlfMean[k]) / prep.hlfStd[k];
      hlf[k] = std::isfinite(value) ? value : 0.f;
    }
  }

  // Standardised features of the packed PF candidates in the cone, without the candidates associated with the
  // photon. Returns the number of candidates; rows holds kPFDim values per candidate.
  int fillPF(const pat::PackedCandidateCollection& candidates,
             const std::unordered_set<std::size_t>& associated,
             const pat::Photon& photon,
             const Preprocessing& prep,
             std::vector<float>& rows) {
    rows.clear();
    const float photonPt = photon.pt();
    const float photonEta = photon.eta();
    const float photonPhi = photon.phi();
    const float photonEnergy = photonPt * std::cosh(photonEta);
    if (!(std::isfinite(photonPt) && photonPt > 0.f && std::isfinite(photonEnergy) && photonEnergy > 0.f))
      return 0;
    int n = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      if (associated.count(i))
        continue;
      const auto& c = candidates[i];
      const float pt = c.pt(), eta = c.eta(), phi = c.phi(), energy = c.energy();
      const float dz = c.dz(), dxy = c.dxy(), mass = c.mass();
      if (!(std::isfinite(pt) && pt >= 0.f && std::isfinite(energy) && energy >= 0.f && std::isfinite(eta) &&
            std::isfinite(phi) && std::isfinite(dz) && std::isfinite(dxy) && std::isfinite(mass)))
        continue;
      const float deta = eta - photonEta;
      const float dphi = wrapPhi(phi - photonPhi);
      const float dr = std::sqrt(deta * deta + dphi * dphi);
      if (!(dr <= prep.pfDrCone))
        continue;
      float features[kPFRawDim] = {std::log(prep.pfLogEps + pt),
                                   eta,
                                   phi,
                                   std::log(prep.pfLogEps + energy),
                                   static_cast<float>(c.pdgId()),
                                   static_cast<float>(c.fromPV()),
                                   dz,
                                   dxy,
                                   std::log(prep.pfLogEps + pt / photonPt),
                                   std::log(prep.pfLogEps + energy / photonEnergy),
                                   deta,
                                   dphi,
                                   dr};
      for (std::size_t k = 0; k < prep.pfStandardizeIdx.size(); ++k) {
        const int index = prep.pfStandardizeIdx[k];
        const float value = (features[index] - prep.pfMean[k]) / prep.pfStd[k];
        features[index] = std::isfinite(value) ? value : 0.f;
      }
      for (float& value : features)
        if (!std::isfinite(value))
          value = 0.f;
      // log pt, eta, phi, log E, one-hot type (6), fromPV, dz, dxy, log pt/pt_photon, log E/E_photon, deta, dphi, dR
      float row[kPFDim] = {};
      std::copy(features, features + 4, row);
      row[4 + pfTypeIndex(c.pdgId())] = 1.f;
      std::copy(features + 5, features + kPFRawDim, row + 10);
      rows.insert(rows.end(), row, row + kPFDim);
      ++n;
    }
    return n;
  }

  // Preshower strips of the photon's endcap near the line from the primary vertex to the supercluster seed, plus
  // the strips of the supercluster's preshower clusters. Returns the number of strips; rows holds kESDim values each.
  int fillES(const pat::Photon& photon,
             const EcalRecHitCollection& esHits,
             const CaloGeometry& geometry,
             const reco::Vertex* vertex,
             const Preprocessing& prep,
             std::vector<float>& rows) {
    rows.clear();
    const reco::SuperCluster& sc = *photon.superCluster();
    if (sc.seed()->seed().subdetId() != EcalEndcap)
      return 0;

    // In MiniAOD the preshower clusters of some superclusters are not stored and the references cannot be resolved.
    // As in training, such photons keep the strips in the window, without links.
    std::unordered_set<uint32_t> linked;
    try {
      for (auto it = sc.preshowerClustersBegin(); it != sc.preshowerClustersEnd(); ++it) {
        if (it->isNull() || !it->isAvailable())
          continue;
        for (const auto& hitAndFraction : (**it).hitsAndFractions())
          linked.insert(hitAndFraction.first.rawId());
      }
    } catch (const cms::Exception& e) {
      if (e.category() != "InvalidReference")
        throw;
      linked.clear();
    }

    const auto& seed = sc.seed()->position();
    const double vx = vertex ? vertex->x() : 0.0, vy = vertex ? vertex->y() : 0.0, vz = vertex ? vertex->z() : 0.0;
    if (std::abs(seed.z() - vz) < 1e-6)
      return 0;
    int n = 0;
    for (const auto& hit : esHits) {
      const DetId id = hit.detid();
      const auto cell = geometry.getGeometry(id);
      if (!cell)
        continue;
      const auto& position = cell->getPosition();
      if (position.z() * seed.z() <= 0)
        continue;
      const double scale = (position.z() - vz) / (seed.z() - vz);
      const double dx = position.x() - (vx + scale * (seed.x() - vx));
      const double dy = position.y() - (vy + scale * (seed.y() - vy));
      const bool inWindow = std::abs(dx) <= prep.esWindowHalfWidthCm && std::abs(dy) <= prep.esWindowHalfWidthCm;
      const bool isLinked = linked.count(id.rawId()) != 0;
      if (!(inWindow || isLinked))
        continue;
      const int plane = ESDetId(id).plane();
      const float energy = hit.energy();
      const float signedLog =
          (energy > 0.f ? 1.f : (energy < 0.f ? -1.f : 0.f)) * std::log1p(std::abs(energy) / prep.esLogScaleGeV);
      float row[kESDim] = {static_cast<float>(dx) / kESOffsetScaleCm,
                           static_cast<float>(dy) / kESOffsetScaleCm,
                           (signedLog - prep.esEnergyMean) / prep.esEnergyStd,
                           plane == 1 ? 1.f : 0.f,
                           plane == 2 ? 1.f : 0.f,
                           isLinked ? 1.f : 0.f};
      for (float& value : row)
        if (!std::isfinite(value))
          value = 0.f;
      rows.insert(rows.end(), row, row + kESDim);
      ++n;
    }
    return n;
  }

}  // namespace

class PhotonTransformerIDProducer : public edm::stream::EDProducer<edm::GlobalCache<photonTransformerID::Cache>> {
public:
  PhotonTransformerIDProducer(const edm::ParameterSet&, const photonTransformerID::Cache*);

  static std::unique_ptr<photonTransformerID::Cache> initializeGlobalCache(const edm::ParameterSet& cfg) {
    return std::make_unique<photonTransformerID::Cache>(cfg);
  }
  static void globalEndJob(const photonTransformerID::Cache*) {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  static void putValueMap(edm::Event&,
                          const edm::Handle<edm::View<pat::Photon>>&,
                          const std::vector<float>&,
                          const std::string&);

  const edm::EDGetTokenT<edm::View<pat::Photon>> photonsToken_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> pfCandidatesToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHitsToken_;
  const edm::EDGetTokenT<EcalRecHitCollection> esRecHitsToken_;
  const edm::EDGetTokenT<double> rhoToken_;
  const edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
  std::vector<std::string> featureNames_;

  // per-photon inputs, reused between events
  std::vector<float> images_, hlf_, geom_;
  std::vector<std::vector<float>> pfRows_, esRows_;
};

PhotonTransformerIDProducer::PhotonTransformerIDProducer(const edm::ParameterSet& cfg,
                                                         const photonTransformerID::Cache* cache)
    : photonsToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
      pfCandidatesToken_(consumes(cfg.getParameter<edm::InputTag>("packedPFCandidates"))),
      ebRecHitsToken_(consumes(cfg.getParameter<edm::InputTag>("ebRecHits"))),
      eeRecHitsToken_(consumes(cfg.getParameter<edm::InputTag>("eeRecHits"))),
      esRecHitsToken_(consumes(cfg.getParameter<edm::InputTag>("esRecHits"))),
      rhoToken_(consumes(cfg.getParameter<edm::InputTag>("rho"))),
      verticesToken_(consumes(cfg.getParameter<edm::InputTag>("vertices"))),
      geometryToken_(esConsumes()) {
  produces<edm::ValueMap<float>>("score");
  for (int k = 0; k < cache->preprocessing.nFeatures; ++k) {
    featureNames_.push_back("feature" + std::to_string(k));
    produces<edm::ValueMap<float>>(featureNames_.back());
  }
}

void PhotonTransformerIDProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  const auto photons = event.getHandle(photonsToken_);
  const auto pfCandidates = event.getHandle(pfCandidatesToken_);
  const auto& ebHits = event.get(ebRecHitsToken_);
  const auto& eeHits = event.get(eeRecHitsToken_);
  const auto& esHits = event.get(esRecHitsToken_);
  const float rho = event.get(rhoToken_);
  const auto& vertices = event.get(verticesToken_);
  const reco::Vertex* primaryVertex = vertices.empty() ? nullptr : &vertices.front();
  const auto& geometry = setup.getData(geometryToken_);
  const photonTransformerID::Preprocessing& prep = globalCache()->preprocessing;

  const int nPhotons = photons->size();
  const int nFeatures = featureNames_.size();
  std::vector<float> scores(nPhotons, -1.f);
  std::vector<std::vector<float>> features(nFeatures, std::vector<float>(nPhotons, 0.f));

  if (nPhotons > 0) {
    images_.resize(nPhotons * kNCells);
    hlf_.resize(nPhotons * kNHLF);
    geom_.resize(nPhotons * 3);
    pfRows_.resize(nPhotons);
    esRows_.resize(nPhotons);
    std::vector<int> nPF(nPhotons), nES(nPhotons);
    std::unordered_set<std::size_t> associated;
    for (int i = 0; i < nPhotons; ++i) {
      const pat::Photon& photon = (*photons)[i];
      fillImage(photon, ebHits, eeHits, prep, &images_[i * kNCells]);
      fillHLF(photon, rho, prep, &hlf_[i * kNHLF]);
      associated.clear();
      for (const auto& ref : photon.associatedPackedPFCandidates())
        if (ref.isNonnull() && ref.id() == pfCandidates.id())
          associated.insert(ref.key());
      nPF[i] = fillPF(*pfCandidates, associated, photon, prep, pfRows_[i]);
      const bool hasPF = nPF[i] > 0;
      if (!hasPF) {
        // as in training: photons without PF candidates get one all-zero candidate that is not masked
        pfRows_[i].assign(kPFDim, 0.f);
        nPF[i] = 1;
      }
      nES[i] = fillES(photon, esHits, geometry, primaryVertex, prep, esRows_[i]);
      // as in training: the photon phi is 0 for photons without PF candidates
      geom_[i * 3 + 0] = photon.eta();
      geom_[i * 3 + 1] = hasPF ? static_cast<float>(photon.phi()) : 0.f;
      geom_[i * 3 + 2] = std::abs(photon.superCluster()->eta()) < 1.479 ? 1.f : 0.f;
    }

    // one batch per event; candidates and strips are padded to the largest number in the event and masked
    const int maxPF = *std::max_element(nPF.begin(), nPF.end());
    const int maxES = *std::max_element(nES.begin(), nES.end());
    const auto floatOptions = torch::TensorOptions().dtype(torch::kFloat32);
    auto image = torch::from_blob(images_.data(), {nPhotons, 1, kGrid, kGrid}, floatOptions);
    auto aux = torch::from_blob(hlf_.data(), {nPhotons, kNHLF}, floatOptions);
    auto geom = torch::from_blob(geom_.data(), {nPhotons, 3}, floatOptions);
    auto pf = torch::zeros({nPhotons, maxPF, kPFDim}, floatOptions);
    auto pfMask = torch::zeros({nPhotons, maxPF}, torch::kBool);
    auto es = torch::zeros({nPhotons, maxES, kESDim}, floatOptions);
    auto esMask = torch::zeros({nPhotons, maxES}, torch::kBool);
    float* pfData = pf.data_ptr<float>();
    float* esData = es.data_ptr<float>();
    for (int i = 0; i < nPhotons; ++i) {
      std::copy(pfRows_[i].begin(), pfRows_[i].end(), pfData + static_cast<std::size_t>(i) * maxPF * kPFDim);
      pfMask[i].slice(0, 0, nPF[i]).fill_(true);
      if (nES[i] > 0) {
        std::copy(esRows_[i].begin(), esRows_[i].end(), esData + static_cast<std::size_t>(i) * maxES * kESDim);
        esMask[i].slice(0, 0, nES[i]).fill_(true);
      }
    }

    std::vector<torch::IValue> inputs{image, aux, pf, pfMask, es, esMask, geom};
    const auto output = globalCache()->model->forward(inputs).toTuple();
    const auto probabilities = torch::sigmoid(output->elements()[0].toTensor()).reshape({-1}).contiguous();
    const auto learned = output->elements()[1].toTensor().contiguous();
    if (probabilities.numel() != nPhotons || learned.dim() != 2 || learned.size(0) != nPhotons ||
        learned.size(1) != nFeatures)
      throw cms::Exception("PhotonTransformerID") << "unexpected model output shapes for " << nPhotons << " photons";
    const float* scoreData = probabilities.data_ptr<float>();
    const float* featureData = learned.data_ptr<float>();
    for (int i = 0; i < nPhotons; ++i) {
      scores[i] = scoreData[i];
      for (int k = 0; k < nFeatures; ++k)
        features[k][i] = featureData[i * nFeatures + k];
    }
  }

  putValueMap(event, photons, scores, "score");
  for (int k = 0; k < nFeatures; ++k)
    putValueMap(event, photons, features[k], featureNames_[k]);
}

void PhotonTransformerIDProducer::putValueMap(edm::Event& event,
                                              const edm::Handle<edm::View<pat::Photon>>& photons,
                                              const std::vector<float>& values,
                                              const std::string& label) {
  auto valueMap = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler filler(*valueMap);
  filler.insert(photons, values.begin(), values.end());
  filler.fill();
  event.put(std::move(valueMap), label);
}

void PhotonTransformerIDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("slimmedPhotons"))->setComment("MiniAOD photons");
  desc.add<edm::InputTag>("packedPFCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("ebRecHits", edm::InputTag("reducedEgamma", "reducedEBRecHits"));
  desc.add<edm::InputTag>("eeRecHits", edm::InputTag("reducedEgamma", "reducedEERecHits"));
  desc.add<edm::InputTag>("esRecHits", edm::InputTag("reducedEgamma", "reducedESRecHits"));
  desc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoAll"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::FileInPath>(
          "modelFile",
          edm::FileInPath("RecoEgamma/PhotonIdentification/data/PhotonTransformerID/Run3_2024_v1/model.pt"))
      ->setComment("TorchScript model");
  desc.add<edm::FileInPath>(
          "preprocessingFile",
          edm::FileInPath("RecoEgamma/PhotonIdentification/data/PhotonTransformerID/Run3_2024_v1/preprocessing.json"))
      ->setComment("standardisation constants and input conventions of the model");
  descriptions.addWithDefaultLabel(desc);
  descriptions.setComment(
      "Transformer photon ID on MiniAOD photons (ECAL crystals, preshower strips and PF candidates). Produces the "
      "ValueMaps 'score' (sigmoid, signal = 1) and 'feature0', 'feature1', ... (learned features). Needs "
      "PyTorchService.");
}

DEFINE_FWK_MODULE(PhotonTransformerIDProducer);

/**
 * Run3ScoutingParticleToPackedCandidateProducer
 *
 * Converts Run3ScoutingParticle to pat::PackedCandidate for MiniAOD compatibility.
 * Matches charged candidates to reco::Tracks to embed track details
 * (hasTrackDetails() == true, dxyError/dzError/normalizedChi2/hit counts available).
 *
 * Requires vertices and scoutingTracks to be produced first.
 */

#include <memory>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Scouting/interface/Run3ScoutingParticle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

class Run3ScoutingParticleToPackedCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit Run3ScoutingParticleToPackedCandidateProducer(const edm::ParameterSet&);
  ~Run3ScoutingParticleToPackedCandidateProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  const edm::EDGetTokenT<std::vector<Run3ScoutingParticle>> particleToken_;
  const edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  const edm::EDGetTokenT<reco::TrackCollection> trackToken_;
  const edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> pdtToken_;
  const bool useCHS_;
  const int covarianceVersion_;
  const int covarianceSchema_;
};

Run3ScoutingParticleToPackedCandidateProducer::Run3ScoutingParticleToPackedCandidateProducer(
    const edm::ParameterSet& iConfig)
    : particleToken_(consumes<std::vector<Run3ScoutingParticle>>(iConfig.getParameter<edm::InputTag>("src"))),
      vertexToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      trackToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      pdtToken_(esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord>()),
      useCHS_(iConfig.getParameter<bool>("CHS")),
      covarianceVersion_(iConfig.getParameter<int>("covarianceVersion")),
      covarianceSchema_(iConfig.getParameter<int>("covarianceSchema")) {
  produces<pat::PackedCandidateCollection>();
}

void Run3ScoutingParticleToPackedCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output = std::make_unique<pat::PackedCandidateCollection>();

  const auto& pdt = iSetup.getData(pdtToken_);
  const auto& particles = iEvent.get(particleToken_);

  auto verticesHandle = iEvent.getHandle(vertexToken_);
  const auto& vertices = *verticesHandle;
  reco::VertexRefProd vertexRefProd(verticesHandle);

  const auto& tracks = iEvent.get(trackToken_);

  // Build a "used" flag so each reco::Track is matched at most once
  std::vector<bool> trackUsed(tracks.size(), false);

  output->reserve(particles.size());

  for (const auto& particle : particles) {
    if (useCHS_ && particle.vertex() > 0) {
      continue;
    }

    const HepPDT::ParticleData* pdtData = pdt.particle(HepPDT::ParticleID(particle.pdgId()));
    if (!pdtData) {
      continue;
    }
    float mass = pdtData->mass().value();

    float pt = particle.pt();
    float eta = particle.eta();
    float phi = particle.phi();
    float px = pt * std::cos(phi);
    float py = pt * std::sin(phi);
    float pz = pt * std::sinh(eta);
    float energy = std::sqrt(px * px + py * py + pz * pz + mass * mass);
    math::XYZTLorentzVector p4(px, py, pz, energy);

    bool relativeTrackVars = particle.relative_trk_vars();
    float trkPt = relativeTrackVars ? particle.trk_pt() + particle.pt() : particle.trk_pt();
    float trkEta = relativeTrackVars ? particle.trk_eta() + particle.eta() : particle.trk_eta();
    float trkPhi = relativeTrackVars ? particle.trk_phi() + particle.phi() : particle.trk_phi();

    int vtxIdx = particle.vertex();
    reco::VertexRef::key_type pvKey = 0;
    if (vtxIdx >= 0 && static_cast<size_t>(vtxIdx) < vertices.size()) {
      pvKey = static_cast<reco::VertexRef::key_type>(vtxIdx);
    }

    math::XYZPoint pvPos(0, 0, 0);
    if (pvKey < vertices.size()) {
      pvPos = vertices[pvKey].position();
    }

    float dxy = particle.dxy();
    float dz = particle.dz();
    float sinPhi = std::sin(phi);
    float cosPhi = std::cos(phi);

    math::XYZPoint vtxPos(pvPos.X() - dxy * sinPhi, pvPos.Y() + dxy * cosPhi, pvPos.Z() + dz);

    pat::PackedCandidate cand(p4, vtxPos, trkPt, trkEta, trkPhi, particle.pdgId(), vertexRefProd, pvKey);

    // Set lost inner hits
    pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
    uint8_t scoutingLostHits = particle.lostInnerHits();
    if (scoutingLostHits == 0) {
      lostHits = pat::PackedCandidate::noLostInnerHits;
    } else if (scoutingLostHits == 1) {
      lostHits = pat::PackedCandidate::oneLostInnerHit;
    } else if (scoutingLostHits >= 2) {
      lostHits = pat::PackedCandidate::moreLostInnerHits;
    }
    cand.setLostInnerHits(lostHits);

    // Set track quality
    int quality = particle.quality();
    bool highPurity = (quality & 4);
    cand.setTrackHighPurity(highPurity);

    // Set PV association quality
    if (vtxIdx == 0) {
      cand.setAssociationQuality(pat::PackedCandidate::UsedInFitTight);
    } else if (vtxIdx > 0) {
      cand.setAssociationQuality(pat::PackedCandidate::CompatibilityDz);
    } else {
      cand.setAssociationQuality(pat::PackedCandidate::NotReconstructedPrimary);
    }

    // Match charged candidates to reco::Tracks and embed track details
    if (particle.pdgId() != 22 && particle.pdgId() != 130 && abs(particle.pdgId()) > 2 && trkPt > 0) {
      int bestIdx = -1;
      float bestMetric = 999.f;
      for (size_t iTk = 0; iTk < tracks.size(); ++iTk) {
        if (trackUsed[iTk])
          continue;
        const auto& tk = tracks[iTk];
        float dEta = trkEta - tk.eta();
        float dPhi = reco::deltaPhi(trkPhi, tk.phi());
        float dR2 = dEta * dEta + dPhi * dPhi;
        float dPtRel = std::abs(trkPt - tk.pt()) / trkPt;
        float metric = dR2 + dPtRel * dPtRel;
        if (metric < bestMetric) {
          bestMetric = metric;
          bestIdx = iTk;
        }
      }
      // Require reasonable match quality
      if (bestIdx >= 0 && bestMetric < 0.01f) {
        cand.setTrackProperties(tracks[bestIdx], covarianceSchema_, covarianceVersion_);
        trackUsed[bestIdx] = true;
      }
    }

    output->push_back(cand);
  }

  iEvent.put(std::move(output));
}

void Run3ScoutingParticleToPackedCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("hltScoutingPFPacker"))
      ->setComment("Input scouting particle collection");
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"))
      ->setComment("Input vertex collection for vertex references");
  desc.add<edm::InputTag>("tracks", edm::InputTag("scoutingTracks"))
      ->setComment("Input reco::Track collection for embedding track details");
  desc.add<bool>("CHS", false)->setComment("Apply Charged Hadron Subtraction (skip vtx > 0)");
  desc.add<int>("covarianceVersion", 1)->setComment("Covariance parameterization version (0=Phase0, 1=Phase1)");
  desc.add<int>("covarianceSchema", 520)->setComment("Covariance packing schema");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(Run3ScoutingParticleToPackedCandidateProducer);

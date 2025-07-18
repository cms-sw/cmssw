#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "CondFormats/PhysicsToolsObjects/interface/DeDxCalibration.h"
#include "CondFormats/DataRecord/interface/DeDxCalibrationRcd.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineService.h"

#include <fstream>

class DeDxHitCalibrator : public edm::stream::EDProducer<> {
public:
  static constexpr int kIsNormal = 0, kIsBelow = 1, kIsOver = 2;
  static constexpr int PXB = 0, PXF = 1, TIB = 2, TID = 3, TOB = 4, TECThin = 5, TECThick = 6;
  typedef std::pair<uint32_t, unsigned char> ChipId;

  explicit DeDxHitCalibrator(const edm::ParameterSet&);
  ~DeDxHitCalibrator() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginRun(edm::Run const&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  int getDetId(const DetId&, const float&);
  float correctEnergy(const float&, const ChipId&);
  void processHitInfo(const reco::DeDxHitInfo&,
                      const float& trackMomentum,
                      reco::DeDxHitCollection&,
                      reco::DeDxHitCollection&);

  double getChi2(const std::vector<double>&, const std::vector<std::pair<double, int> >&, const double&, const double&);
  void getAlphaBeta(const std::vector<double>&,
                    const std::vector<std::pair<double, int> >&,
                    CLHEP::HepMatrix&,
                    CLHEP::HepVector&,
                    const std::vector<bool>&,
                    const double&,
                    const double&);
  std::pair<double, double> fitStripCluster(const std::vector<std::pair<double, int> >&, const double&, const double&);

  const bool applyGain_;
  const double MeVPerElectron_;
  const int VCaltoElectronGain_, VCaltoElectronGain_L1_, VCaltoElectronOffset_, VCaltoElectronOffset_L1_;
  const int pixelSaturationThr_;
  const edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  const edm::EDGetTokenT<reco::DeDxHitInfoAss> dedxHitInfoToken_;
  const edm::ESGetToken<DeDxCalibration, DeDxCalibrationRcd> dedxCalibToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;

  SiPixelGainCalibrationOfflineService pixelCalib_;
  edm::ESHandle<DeDxCalibration> dedxCalib_;
  edm::ESHandle<TrackerGeometry> tkGeom_;
  edm::ESHandle<TrackerTopology> tkTopo_;
};

DeDxHitCalibrator::DeDxHitCalibrator(const edm::ParameterSet& iConfig)
    : applyGain_(iConfig.getParameter<bool>("applyGain")),
      MeVPerElectron_(iConfig.getParameter<double>("MeVPerElectron")),
      VCaltoElectronGain_(iConfig.getParameter<int>("VCaltoElectronGain")),
      VCaltoElectronGain_L1_(iConfig.getParameter<int>("VCaltoElectronGain_L1")),
      VCaltoElectronOffset_(iConfig.getParameter<int>("VCaltoElectronOffset")),
      VCaltoElectronOffset_L1_(iConfig.getParameter<int>("VCaltoElectronOffset_L1")),
      pixelSaturationThr_(iConfig.getParameter<int>("pixelSaturationThr")),
      tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("trackProducer"))),
      dedxHitInfoToken_(consumes<reco::DeDxHitInfoAss>(iConfig.getParameter<edm::InputTag>("dedxHitInfo"))),
      dedxCalibToken_(esConsumes<DeDxCalibration, DeDxCalibrationRcd, edm::Transition::BeginRun>()),
      tkGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      tkTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      pixelCalib_(iConfig, consumesCollector()) {
  produces<reco::TrackDeDxHitsCollection>("PixelHits");
  produces<reco::TrackDeDxHitsCollection>("StripHits");
}

void DeDxHitCalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("applyGain", true);
  desc.add<double>("MeVPerElectron", 3.61e-06);
  desc.add<int>("VCaltoElectronGain", 65);
  desc.add<int>("VCaltoElectronGain_L1", 65);
  desc.add<int>("VCaltoElectronOffset", -414);
  desc.add<int>("VCaltoElectronOffset_L1", -414);
  desc.add<int>("pixelSaturationThr", 254);
  desc.add<edm::InputTag>("trackProducer", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("dedxHitInfo", edm::InputTag("dedxHitInfo"));
  descriptions.add("dedxHitCalibrator", desc);
}

void DeDxHitCalibrator::beginRun(edm::Run const&, const edm::EventSetup& iSetup) {
  dedxCalib_ = iSetup.getHandle(dedxCalibToken_);
  tkGeom_ = iSetup.getHandle(tkGeomToken_);
  tkTopo_ = iSetup.getHandle(tkTopoToken_);
}

void DeDxHitCalibrator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& tracks = iEvent.getHandle(tracksToken_);
  const auto& dedxHitInfo = iEvent.get(dedxHitInfoToken_);
  pixelCalib_.setESObjects(iSetup);

  // creates the output collection
  auto pixelTrackDeDxHitAss = std::make_unique<reco::TrackDeDxHitsCollection>(reco::TrackRefProd(tracks));
  auto stripTrackDeDxHitAss = std::make_unique<reco::TrackDeDxHitsCollection>(reco::TrackRefProd(tracks));

  for (size_t i = 0; i < tracks->size(); i++) {
    const auto& track = reco::TrackRef(tracks, i);
    const auto& dedxHits = dedxHitInfo[track];
    if (dedxHits.isNull())
      continue;
    reco::DeDxHitCollection pixelHits, stripHits;
    processHitInfo(*dedxHits, track->p(), pixelHits, stripHits);
    pixelTrackDeDxHitAss->setValue(i, pixelHits);
    stripTrackDeDxHitAss->setValue(i, stripHits);
  }

  iEvent.put(std::move(pixelTrackDeDxHitAss), "PixelHits");
  iEvent.put(std::move(stripTrackDeDxHitAss), "StripHits");
}

/*****************************************************************************/
void DeDxHitCalibrator::processHitInfo(const reco::DeDxHitInfo& info,
                                       const float& trackMomentum,
                                       reco::DeDxHitCollection& pixelHits,
                                       reco::DeDxHitCollection& stripHits) {
  // Energy loss parametrization from https://doi.org/10.1016/j.nima.2012.06.064
  static constexpr double a = 0.07;     //energy loss factor in silicon
  static constexpr double l0 = 450e-4;  //reference path length in um

  for (size_t i = 0; i < info.size(); i++) {
    // Require hits to be complete and compatible
    const auto& type = info.type(i);
    if (!(type & (1 << reco::DeDxHitInfo::Complete)) || !(type & (1 << reco::DeDxHitInfo::Compatible)))
      continue;

    // Effective path length
    const auto& pl = info.pathlength(i);
    const auto pathLength = pl * (1. + a * std::log(pl / l0));
    const DetId& detId = info.detId(i);

    // Strip
    if (const auto& stripCluster = info.stripCluster(i)) {
      const auto& thickness =
          dynamic_cast<const StripGeomDetUnit*>(tkGeom_->idToDet(detId))->surface().bounds().thickness();
      const auto& det = getDetId(detId, thickness);
      const auto& chipId = ChipId(detId, stripCluster->barycenter() / sistrip::STRIPS_PER_APV);
      // Fill measured strip deposits
      const auto& thr = dedxCalib_->thr()[det];
      std::vector<std::pair<double, int> > b;
      b.reserve(stripCluster->amplitudes().size() + 2);
      b.emplace_back(thr, kIsBelow);
      for (const auto& adc : stripCluster->amplitudes()) {
        if (adc > 253)
          b.emplace_back(254., kIsOver);
        else
          b.emplace_back(adc + 0.5, kIsNormal);
      }
      b.emplace_back(thr, kIsBelow);
      // Fit
      const auto& result = fitStripCluster(b, dedxCalib_->alpha()[det], 1. / dedxCalib_->sigma()[det]);
      // ADC -> e- -> MeV
      const auto& energy = correctEnergy(result.first * sistrip::MeVperADCStrip, chipId);
      const auto& esigma = correctEnergy(result.second * sistrip::MeVperADCStrip, chipId);
      // Compute charge
      const auto& charge = pathLength != 0. ? energy / pathLength : 0.;
      stripHits.emplace_back(charge, trackMomentum, pathLength, 0, esigma);
    }

    // Pixel
    if (const auto& pixelCluster = info.pixelCluster(i)) {
      const auto& thickness =
          dynamic_cast<const PixelGeomDetUnit*>(tkGeom_->idToDet(detId))->surface().bounds().thickness();
      const auto& det = getDetId(detId, thickness);
      const auto& chipId =
          ChipId(detId, (int(pixelCluster->x() / ROCSizeInX) << 3) + int(pixelCluster->y() / ROCSizeInY));
      // Collect adc
      double delta(0);
      bool isSaturated(false);
      for (size_t j = 0; j < pixelCluster->pixelADC().size(); j++) {
        const auto& elec = pixelCluster->pixelADC()[j];
        delta += elec * MeVPerElectron_;
        if (isSaturated)
          continue;
        // ( pos , (x , y) )
        const auto& row = pixelCluster->minPixelRow() + pixelCluster->pixelOffset()[2 * j];
        const auto& col = pixelCluster->minPixelCol() + pixelCluster->pixelOffset()[2 * j + 1];
        // Go back to adc
        const auto& DBgain = pixelCalib_.getGain(detId, col, row);
        const auto& DBpedestal = pixelCalib_.getPedestal(detId, col, row);
        if (elec == std::numeric_limits<uint16_t>::max())
          isSaturated = true;
        else if (DBgain > 0.) {
          double vcal;
          const auto& theLayer = (detId.subdetId() == 1) ? tkTopo_->pxbLayer(detId) : 0;
          if (theLayer == 1)
            vcal = (elec - VCaltoElectronOffset_L1_) / VCaltoElectronGain_L1_;
          else
            vcal = (elec - VCaltoElectronOffset_) / VCaltoElectronGain_;
          const auto adc = std::round(DBpedestal + vcal / DBgain);
          if (adc > pixelSaturationThr_)
            isSaturated = true;
        }
      }
      // Compute energy
      const auto& energy = correctEnergy(delta, chipId);
      if (det != PXF && energy <= 50e-3)
        continue;
      // Estimate sigma for cluster
      const unsigned char& nChannels = pixelCluster->pixelADC().size();
      const auto esigma = 10e-3 * std::sqrt(nChannels);
      // Compute charge
      const auto& charge = pathLength != 0. ? energy / pathLength : 0.;
      pixelHits.emplace_back(charge, trackMomentum, pathLength, isSaturated, esigma);
    }
  }
}

/*****************************************************************************/
int DeDxHitCalibrator::getDetId(const DetId& id, const float& thickness) {
  const auto& subdet = id.subdetId() - 1;
  if (subdet == TECThin && thickness > 400e-4)
    return TECThick;
  return subdet;
}

/*****************************************************************************/
float DeDxHitCalibrator::correctEnergy(const float& energy, const ChipId& detId) {
  const auto& g = dedxCalib_->gain().find(detId);
  if (applyGain_ && g != dedxCalib_->gain().end())
    return energy * g->second;
  return energy;
}

/*****************************************************************************/
double DeDxHitCalibrator::getChi2(const std::vector<double>& x,
                                  const std::vector<std::pair<double, int> >& b,
                                  const double& coupling,
                                  const double& iSigma) {
  const auto& npar = b.size();
  std::vector<double> y(npar, 0.);
  for (size_t i = 0; i < npar; i++) {
    const auto dx = coupling * x[i];
    if (i >= 1)
      y[i - 1] += dx;
    y[i] += x[i] - 2 * dx;
    if (i + 1 < npar)
      y[i + 1] += dx;
  }
  double chi2(0.);
  for (size_t i = 0; i < npar; i++) {
    auto q = (b[i].first - y[i]) * iSigma;
    if (b[i].second == kIsNormal)
      chi2 += q * q;
    else if (b[i].second == kIsBelow) {
      q -= 2;
      if (q < 0)
        chi2 += 0.5 * q * q;
    } else if (b[i].second == kIsOver) {
      q += 2;
      if (q > 0)
        chi2 += 0.5 * q * q;
    }
    // Penalty for negatives
    if (x[i] < 0) {
      q = x[i] * iSigma;
      chi2 += q * q;
    }
  }
  return chi2;
}

/*****************************************************************************/
void DeDxHitCalibrator::getAlphaBeta(const std::vector<double>& x,
                                     const std::vector<std::pair<double, int> >& b,
                                     CLHEP::HepMatrix& alpha,
                                     CLHEP::HepVector& beta,
                                     const std::vector<bool>& isFix,
                                     const double& coupling,
                                     const double& iSigma) {
  const auto& npar = b.size();
  const auto a0 = coupling * iSigma;
  const auto a1 = (1 - 2 * coupling) * iSigma;
  const auto a00 = a0 * a0;
  const auto a11 = a1 * a1;
  const auto a01 = a0 * a1;
  std::vector<double> y(npar, 0.);
  for (size_t i = 0; i < npar; i++)
    if (!isFix[i]) {
      const auto dx = coupling * x[i];
      if (i >= 1)
        y[i - 1] += dx;
      y[i] += x[i] - 2 * dx;
      if (i + 1 < npar)
        y[i + 1] += dx;
    }
  for (size_t i = 0; i < npar; i++) {
    auto q = (y[i] - b[i].first) * iSigma;
    int f(0);
    if (b[i].second == kIsNormal)
      f = 2;
    else if (b[i].second == kIsBelow) {
      q += 2;
      if (q > 0)
        f = 1;
    } else if (b[i].second == kIsOver) {
      q -= 2;
      if (q < 0)
        f = 1;
    }
    if (f > 0) {
      if (i >= 1)
        if (!isFix[i - 1]) {
          alpha[i - 1][i - 1] += f * a00;
          if (!isFix[i]) {
            alpha[i - 1][i] += f * a01;
            alpha[i][i - 1] += f * a01;
          }
          beta[i - 1] += f * q * a0;
        }
      if (!isFix[i]) {
        alpha[i][i] += f * a11;
        beta[i] += f * q * a1;
      }
      if (i + 1 < npar)
        if (!isFix[i + 1]) {
          alpha[i + 1][i + 1] += f * a00;
          if (!isFix[i]) {
            alpha[i + 1][i] += f * a01;
            alpha[i][i + 1] += f * a01;
          }
          beta[i + 1] += f * q * a0;
        }
    }
    // Penalty for negatives
    if (!isFix[i])
      if (x[i] < 0) {
        alpha[i][i] += 2 * iSigma * iSigma;
        q = x[i] * iSigma;
        beta[i] += 2 * q * iSigma;
      }
  }
  for (size_t i = 0; i < npar; i++)
    if (isFix[i]) {
      alpha[i][i] = 1.;
      beta[i] = 0.;
    }
}

/*****************************************************************************/
std::pair<double, double> DeDxHitCalibrator::fitStripCluster(const std::vector<std::pair<double, int> >& b,
                                                             const double& coupling,
                                                             const double& iSigma) {
  const auto& npar = b.size();
  std::vector<bool> isFix(npar, false);
  std::vector<double> x;
  x.reserve(b.size());
  for (const auto& ib : b)
    x.emplace_back(ib.first);

  bool ok = false;
  CLHEP::HepMatrix hessian(npar, npar);
  int iter = 0;
  do {
    double diff(0);
    auto old = getChi2(x, b, coupling, iSigma);
    do {
      CLHEP::HepMatrix alpha(npar, npar, 0.);
      CLHEP::HepVector beta(npar, 0.);
      getAlphaBeta(x, b, alpha, beta, isFix, coupling, iSigma);
      const auto& delta = CLHEP::solve(alpha, -beta);
      double lambda(1.);
      std::vector<double> xn(npar);
      for (size_t i = 0; i < npar; i++)
        xn[i] = x[i] + lambda * delta[i];
      const auto& next = getChi2(xn, b, coupling, iSigma);
      diff = old - next;
      if (diff > 0)
        x = xn;
      old = next;
      hessian = alpha;
      iter++;
    } while (diff > 1e-6 && iter < 100);
    // Check if we have negatives
    const auto& mi = std::min_element(x.begin(), x.end()) - x.begin();
    if (x[mi] < 0) {
      x[mi] = 0.;
      isFix[mi] = true;
    } else
      ok = true;
  } while (!ok && iter < 100);
  hessian /= 2.;
  int flag;
  hessian.invert(flag);

  double var2(0.);
  for (size_t i = 0; i < npar; i++)
    for (size_t j = 0; j < npar; j++)
      if (!isFix[i] && !isFix[j])
        var2 += hessian[i][j];
  var2 = std::abs(var2);

  double sum(0.);
  for (size_t i = 0; i < npar; i++)
    if (!isFix[i])
      sum += x[i];

  return {sum, std::sqrt(var2)};
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeDxHitCalibrator);

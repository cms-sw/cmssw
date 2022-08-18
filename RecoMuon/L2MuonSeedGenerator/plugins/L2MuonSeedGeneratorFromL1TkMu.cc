/*  \class L2MuonSeedGeneratorFromL1TkMu
 *
 *   L2 muon seed generator:
 *   Transform the L1TkMuon informations in seeds
 *   for the L2 muon reconstruction
 *   (mimicking L2MuonSeedGeneratorFromL1T)
 *
 *    Author: H. Kwon
 *    Modified by M. Oh
 */

// Framework
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

// Data Formats
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L2MuonTrajectorySeedCollection.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "Geometry/CommonDetUnit/interface/GeomDetEnumerators.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

using namespace std;
using namespace edm;
using namespace l1t;

class L2MuonSeedGeneratorFromL1TkMu : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit L2MuonSeedGeneratorFromL1TkMu(const edm::ParameterSet &);

  /// Destructor
  ~L2MuonSeedGeneratorFromL1TkMu() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  edm::InputTag source_;
  edm::EDGetTokenT<l1t::TrackerMuonCollection> muCollToken_;

  edm::InputTag offlineSeedLabel_;
  edm::EDGetTokenT<edm::View<TrajectorySeed>> offlineSeedToken_;

  std::string propagatorName_;

  const double l1MinPt_;
  const double l1MaxEta_;
  const double minPtBarrel_;
  const double minPtEndcap_;
  const double minPL1Tk_;
  const double minPtL1TkBarrel_;
  const bool useOfflineSeed_;
  const bool useUnassociatedL1_;
  std::vector<double> matchingDR_;
  std::vector<double> etaBins_;

  // parameters used in propagating L1 tracker track
  // to the second muon station, numbers are
  // taken from L1TkMuonProducer::propagateToGMT
  static constexpr float etaBoundary_{1.1};
  static constexpr float distMB2_{550.};
  static constexpr float distME2_{850.};
  static constexpr float phiCorr0_{1.464};
  static constexpr float phiCorr1_{1.7};
  static constexpr float phiCorr2_{144.};

  /// the event setup proxy, it takes care the services update
  std::unique_ptr<MuonServiceProxy> service_;
  std::unique_ptr<MeasurementEstimator> estimator_;

  const TrajectorySeed *associateOfflineSeedToL1(edm::Handle<edm::View<TrajectorySeed>> &,
                                                 std::vector<int> &,
                                                 TrajectoryStateOnSurface &,
                                                 double);
};

// constructors
L2MuonSeedGeneratorFromL1TkMu::L2MuonSeedGeneratorFromL1TkMu(const edm::ParameterSet &iConfig)
    : source_(iConfig.getParameter<InputTag>("InputObjects")),
      muCollToken_(consumes(source_)),
      propagatorName_(iConfig.getParameter<string>("Propagator")),
      l1MinPt_(iConfig.getParameter<double>("L1MinPt")),
      l1MaxEta_(iConfig.getParameter<double>("L1MaxEta")),
      minPtBarrel_(iConfig.getParameter<double>("SetMinPtBarrelTo")),
      minPtEndcap_(iConfig.getParameter<double>("SetMinPtEndcapTo")),
      minPL1Tk_(iConfig.getParameter<double>("MinPL1Tk")),
      minPtL1TkBarrel_(iConfig.getParameter<double>("MinPtL1TkBarrel")),
      useOfflineSeed_(iConfig.getUntrackedParameter<bool>("UseOfflineSeed", false)),
      useUnassociatedL1_(iConfig.getParameter<bool>("UseUnassociatedL1")),
      matchingDR_(iConfig.getParameter<std::vector<double>>("MatchDR")),
      etaBins_(iConfig.getParameter<std::vector<double>>("EtaMatchingBins")) {
  if (useOfflineSeed_) {
    offlineSeedLabel_ = iConfig.getUntrackedParameter<InputTag>("OfflineSeedLabel");
    offlineSeedToken_ = consumes<edm::View<TrajectorySeed>>(offlineSeedLabel_);

    // check that number of eta bins -1 matches number of dR cones
    if (matchingDR_.size() != etaBins_.size() - 1) {
      throw cms::Exception("Configuration") << "Size of MatchDR "
                                            << "does not match number of eta bins." << endl;
    }
  }

  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");

  // the services
  service_ = std::make_unique<MuonServiceProxy>(serviceParameters, consumesCollector());

  // the estimator
  estimator_ = std::make_unique<Chi2MeasurementEstimator>(10000.);

  produces<L2MuonTrajectorySeedCollection>();
}

void L2MuonSeedGeneratorFromL1TkMu::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects", edm::InputTag("hltGmtStage2Digis"));
  desc.add<string>("Propagator", "");
  desc.add<double>("L1MinPt", -1.);
  desc.add<double>("L1MaxEta", 5.0);
  desc.add<double>("SetMinPtBarrelTo", 3.5);
  desc.add<double>("SetMinPtEndcapTo", 1.0);
  desc.add<double>("MinPL1Tk", 3.5);
  desc.add<double>("MinPtL1TkBarrel", 3.5);
  desc.addUntracked<bool>("UseOfflineSeed", false);
  desc.add<bool>("UseUnassociatedL1", true);
  desc.add<std::vector<double>>("MatchDR", {0.3});
  desc.add<std::vector<double>>("EtaMatchingBins", {0., 2.5});
  desc.addUntracked<edm::InputTag>("OfflineSeedLabel", edm::InputTag(""));

  edm::ParameterSetDescription psd0;
  psd0.addUntracked<std::vector<std::string>>("Propagators", {"SteppingHelixPropagatorAny"});
  psd0.add<bool>("RPCLayers", true);
  psd0.addUntracked<bool>("UseMuonNavigation", true);
  desc.add<edm::ParameterSetDescription>("ServiceParameters", psd0);
  descriptions.add("L2MuonSeedGeneratorFromL1TkMu", desc);
}

void L2MuonSeedGeneratorFromL1TkMu::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const std::string metname = "Muon|RecoMuon|L2MuonSeedGeneratorFromL1TkMu";
  MuonPatternRecoDumper debug;

  auto output = std::make_unique<L2MuonTrajectorySeedCollection>();

  auto const muColl = iEvent.getHandle(muCollToken_);
  LogDebug(metname) << "Number of muons " << muColl->size() << endl;

  edm::Handle<edm::View<TrajectorySeed>> offlineSeedHandle;
  vector<int> offlineSeedMap;
  if (useOfflineSeed_) {
    iEvent.getByToken(offlineSeedToken_, offlineSeedHandle);
    offlineSeedMap = vector<int>(offlineSeedHandle->size(), 0);
  }

  for (auto const &tkmu : *muColl) {
    // L1 tracker track
    auto const &it = tkmu.trkPtr();

    // propagate the L1 tracker track to GMT
    auto p3 = it->momentum();

    float tk_p = p3.mag();
    if (tk_p < minPL1Tk_)
      continue;

    float tk_pt = p3.perp();
    float tk_eta = p3.eta();
    float tk_aeta = std::abs(tk_eta);

    bool barrel = tk_aeta < etaBoundary_;
    if (barrel && tk_pt < minPtL1TkBarrel_)
      continue;

    float tk_phi = p3.phi();
    float tk_q = it->rInv() > 0 ? 1. : -1.;
    float tk_z = it->POCA().z();

    float dzCorrPhi = 1.;
    float deta = 0;
    float etaProp = tk_aeta;

    if (barrel) {
      etaProp = etaBoundary_;
      deta = tk_z / distMB2_ / cosh(tk_aeta);
    } else {
      float delta = tk_z / distME2_;  //roughly scales as distance to 2nd station
      if (tk_eta > 0)
        delta *= -1;
      dzCorrPhi = 1. + delta;

      float zOzs = tk_z / distME2_;
      if (tk_eta > 0)
        deta = zOzs / (1. - zOzs);
      else
        deta = zOzs / (1. + zOzs);
      deta = deta * tanh(tk_eta);
    }
    float resPhi = tk_phi - phiCorr0_ * tk_q * cosh(phiCorr1_) / cosh(etaProp) / tk_pt * dzCorrPhi - M_PI / phiCorr2_;
    resPhi = reco::reduceRange(resPhi);

    float pt = tk_pt;  //not corrected for eloss
    float eta = tk_eta + deta;
    float theta = 2 * atan(exp(-eta));
    float phi = resPhi;
    int charge = it->rInv() > 0 ? 1 : -1;

    if (pt < l1MinPt_ || std::abs(eta) > l1MaxEta_)
      continue;

    LogDebug(metname) << "New L2 Muon Seed";
    LogDebug(metname) << "Pt = " << pt << " GeV/c";
    LogDebug(metname) << "eta = " << eta;
    LogDebug(metname) << "theta = " << theta << " rad";
    LogDebug(metname) << "phi = " << phi << " rad";
    LogDebug(metname) << "charge = " << charge;
    LogDebug(metname) << "In Barrel? = " << barrel;

    // Update the services
    service_->update(iSetup);

    const DetLayer *detLayer = nullptr;
    float radius = 0.;

    CLHEP::Hep3Vector vec(0., 1., 0.);
    vec.setTheta(theta);
    vec.setPhi(phi);

    DetId theid;
    // Get the det layer on which the state should be put
    if (barrel) {
      LogDebug(metname) << "The seed is in the barrel";

      // MB2
      theid = DTChamberId(0, 2, 0);
      detLayer = service_->detLayerGeometry()->idToLayer(theid);
      LogDebug(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

      const BoundSurface *sur = &(detLayer->surface());
      const BoundCylinder *bc = dynamic_cast<const BoundCylinder *>(sur);

      radius = std::abs(bc->radius() / sin(theta));

      LogDebug(metname) << "radius " << radius;

      if (pt < minPtBarrel_)
        pt = minPtBarrel_;
    } else {
      LogDebug(metname) << "The seed is in the endcap";

      // ME2
      theid = theta < Geom::pi() / 2. ? CSCDetId(1, 2, 0, 0, 0) : CSCDetId(2, 2, 0, 0, 0);

      detLayer = service_->detLayerGeometry()->idToLayer(theid);
      LogDebug(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

      radius = std::abs(detLayer->position().z() / cos(theta));

      if (pt < minPtEndcap_)
        pt = minPtEndcap_;
    }

    vec.setMag(radius);

    GlobalPoint pos(vec.x(), vec.y(), vec.z());

    GlobalVector mom(pt * cos(phi), pt * sin(phi), pt * cos(theta) / sin(theta));

    GlobalTrajectoryParameters param(pos, mom, charge, &*service_->magneticField());
    AlgebraicSymMatrix55 mat;

    mat[0][0] = (0.25 / pt) * (0.25 / pt);  // sigma^2(charge/abs_momentum)
    if (!barrel)
      mat[0][0] = (0.4 / pt) * (0.4 / pt);

    mat[1][1] = 0.05 * 0.05;  // sigma^2(lambda)
    mat[2][2] = 0.2 * 0.2;    // sigma^2(phi)
    mat[3][3] = 20. * 20.;    // sigma^2(x_transverse))
    mat[4][4] = 20. * 20.;    // sigma^2(y_transverse))

    CurvilinearTrajectoryError error(mat);

    const FreeTrajectoryState state(param, error);

    LogDebug(metname) << "Free trajectory State from the parameters";
    LogDebug(metname) << debug.dumpFTS(state);

    // Propagate the state on the MB2/ME2 surface
    TrajectoryStateOnSurface tsos = service_->propagator(propagatorName_)->propagate(state, detLayer->surface());

    LogDebug(metname) << "State after the propagation on the layer";
    LogDebug(metname) << debug.dumpLayer(detLayer);
    LogDebug(metname) << debug.dumpTSOS(tsos);

    double dRcone = matchingDR_[0];
    if (std::abs(eta) < etaBins_.back()) {
      std::vector<double>::iterator lowEdge = std::upper_bound(etaBins_.begin(), etaBins_.end(), std::abs(eta));
      dRcone = matchingDR_.at(lowEdge - etaBins_.begin() - 1);
    }

    if (tsos.isValid()) {
      edm::OwnVector<TrackingRecHit> container;

      if (useOfflineSeed_) {
        // Get the compatible dets on the layer
        std::vector<pair<const GeomDet *, TrajectoryStateOnSurface>> detsWithStates =
            detLayer->compatibleDets(tsos, *service_->propagator(propagatorName_), *estimator_);

        if (detsWithStates.empty() && barrel) {
          // Fallback solution using ME2, try again to propagate but using ME2 as reference
          DetId fallback_id;
          theta < Geom::pi() / 2. ? fallback_id = CSCDetId(1, 2, 0, 0, 0) : fallback_id = CSCDetId(2, 2, 0, 0, 0);
          const DetLayer *ME2DetLayer = service_->detLayerGeometry()->idToLayer(fallback_id);

          tsos = service_->propagator(propagatorName_)->propagate(state, ME2DetLayer->surface());
          detsWithStates = ME2DetLayer->compatibleDets(tsos, *service_->propagator(propagatorName_), *estimator_);
        }

        if (!detsWithStates.empty()) {
          TrajectoryStateOnSurface newTSOS = detsWithStates.front().second;
          const GeomDet *newTSOSDet = detsWithStates.front().first;

          LogDebug(metname) << "Most compatible det";
          LogDebug(metname) << debug.dumpMuonId(newTSOSDet->geographicalId());

          if (newTSOS.isValid()) {
            LogDebug(metname) << "pos: (r=" << newTSOS.globalPosition().mag()
                              << ", phi=" << newTSOS.globalPosition().phi()
                              << ", eta=" << newTSOS.globalPosition().eta() << ")";
            LogDebug(metname) << "mom: (q*pt=" << newTSOS.charge() * newTSOS.globalMomentum().perp()
                              << ", phi=" << newTSOS.globalMomentum().phi()
                              << ", eta=" << newTSOS.globalMomentum().eta() << ")";

            const TrajectorySeed *assoOffseed =
                associateOfflineSeedToL1(offlineSeedHandle, offlineSeedMap, newTSOS, dRcone);

            if (assoOffseed != nullptr) {
              PTrajectoryStateOnDet const &seedTSOS = assoOffseed->startingState();
              for (auto const &recHit : assoOffseed->recHits()) {
                container.push_back(recHit);
              }
              auto dummyRef = edm::Ref<MuonBxCollection>();
              output->emplace_back(L2MuonTrajectorySeed(seedTSOS, container, alongMomentum, dummyRef));
            } else {
              if (useUnassociatedL1_) {
                // convert the TSOS into a PTSOD
                PTrajectoryStateOnDet const &seedTSOS =
                    trajectoryStateTransform::persistentState(newTSOS, newTSOSDet->geographicalId().rawId());
                auto dummyRef = edm::Ref<MuonBxCollection>();
                output->emplace_back(L2MuonTrajectorySeed(seedTSOS, container, alongMomentum, dummyRef));
              }
            }
          }
        }
      } else {
        // convert the TSOS into a PTSOD
        PTrajectoryStateOnDet const &seedTSOS = trajectoryStateTransform::persistentState(tsos, theid.rawId());
        auto dummyRef = edm::Ref<MuonBxCollection>();
        output->emplace_back(L2MuonTrajectorySeed(seedTSOS, container, alongMomentum, dummyRef));
      }
    }
  }

  iEvent.put(std::move(output));
}

// FIXME: does not resolve ambiguities yet!
const TrajectorySeed *L2MuonSeedGeneratorFromL1TkMu::associateOfflineSeedToL1(
    edm::Handle<edm::View<TrajectorySeed>> &offseeds,
    std::vector<int> &offseedMap,
    TrajectoryStateOnSurface &newTsos,
    double dRcone) {
  if (dRcone < 0.)
    return nullptr;

  const std::string metlabel = "Muon|RecoMuon|L2MuonSeedGeneratorFromL1TkMu";
  MuonPatternRecoDumper debugtmp;

  edm::View<TrajectorySeed>::const_iterator offseed, endOffseed = offseeds->end();
  const TrajectorySeed *selOffseed = nullptr;
  double bestDr2 = 99999.;
  unsigned int nOffseed(0);
  int lastOffseed(-1);

  for (offseed = offseeds->begin(); offseed != endOffseed; ++offseed, ++nOffseed) {
    if (offseedMap[nOffseed] != 0)
      continue;
    GlobalPoint glbPos = service_->trackingGeometry()
                             ->idToDet(offseed->startingState().detId())
                             ->surface()
                             .toGlobal(offseed->startingState().parameters().position());
    GlobalVector glbMom = service_->trackingGeometry()
                              ->idToDet(offseed->startingState().detId())
                              ->surface()
                              .toGlobal(offseed->startingState().parameters().momentum());

    // Preliminary check
    double preDr2 = deltaR2(newTsos.globalPosition().eta(), newTsos.globalPosition().phi(), glbPos.eta(), glbPos.phi());
    if (preDr2 > 1.0)
      continue;

    const FreeTrajectoryState offseedFTS(
        glbPos, glbMom, offseed->startingState().parameters().charge(), &*service_->magneticField());
    TrajectoryStateOnSurface offseedTsos =
        service_->propagator(propagatorName_)->propagate(offseedFTS, newTsos.surface());
    LogDebug(metlabel) << "Offline seed info: Det and State" << std::endl;
    LogDebug(metlabel) << debugtmp.dumpMuonId(offseed->startingState().detId()) << std::endl;
    LogDebug(metlabel) << "pos: (r=" << offseedFTS.position().mag() << ", phi=" << offseedFTS.position().phi()
                       << ", eta=" << offseedFTS.position().eta() << ")" << std::endl;
    LogDebug(metlabel) << "mom: (q*pt=" << offseedFTS.charge() * offseedFTS.momentum().perp()
                       << ", phi=" << offseedFTS.momentum().phi() << ", eta=" << offseedFTS.momentum().eta() << ")"
                       << std::endl
                       << std::endl;

    if (offseedTsos.isValid()) {
      LogDebug(metlabel) << "Offline seed info after propagation to L1 layer:" << std::endl;
      LogDebug(metlabel) << "pos: (r=" << offseedTsos.globalPosition().mag()
                         << ", phi=" << offseedTsos.globalPosition().phi()
                         << ", eta=" << offseedTsos.globalPosition().eta() << ")" << std::endl;
      LogDebug(metlabel) << "mom: (q*pt=" << offseedTsos.charge() * offseedTsos.globalMomentum().perp()
                         << ", phi=" << offseedTsos.globalMomentum().phi()
                         << ", eta=" << offseedTsos.globalMomentum().eta() << ")" << std::endl
                         << std::endl;
      double newDr2 = deltaR2(newTsos.globalPosition().eta(),
                              newTsos.globalPosition().phi(),
                              offseedTsos.globalPosition().eta(),
                              offseedTsos.globalPosition().phi());
      LogDebug(metlabel) << "   -- DR = " << newDr2 << std::endl;
      if (newDr2 < bestDr2 && newDr2 < dRcone * dRcone) {
        LogDebug(metlabel) << "          --> OK! " << newDr2 << std::endl << std::endl;
        selOffseed = &*offseed;
        bestDr2 = newDr2;
        offseedMap[nOffseed] = 1;
        if (lastOffseed > -1)
          offseedMap[lastOffseed] = 0;
        lastOffseed = nOffseed;
      } else {
        LogDebug(metlabel) << "          --> Rejected. " << newDr2 << std::endl << std::endl;
      }
    } else {
      LogDebug(metlabel) << "Invalid offline seed TSOS after propagation!" << std::endl << std::endl;
    }
  }

  return selOffseed;
}

//define this as a plug-in
DEFINE_FWK_MODULE(L2MuonSeedGeneratorFromL1TkMu);

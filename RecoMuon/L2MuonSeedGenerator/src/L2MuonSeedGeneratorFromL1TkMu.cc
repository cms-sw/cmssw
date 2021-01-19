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

// Class Header
#include "RecoMuon/L2MuonSeedGenerator/src/L2MuonSeedGeneratorFromL1TkMu.h"

// Framework
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;
using namespace edm;
using namespace l1t;

// constructors
L2MuonSeedGeneratorFromL1TkMu::L2MuonSeedGeneratorFromL1TkMu(const edm::ParameterSet &iConfig)
    : theSource(iConfig.getParameter<InputTag>("InputObjects")),
      thePropagatorName(iConfig.getParameter<string>("Propagator")),
      theL1MinPt(iConfig.getParameter<double>("L1MinPt")),
      theL1MaxEta(iConfig.getParameter<double>("L1MaxEta")),
      theMinPtBarrel(iConfig.getParameter<double>("SetMinPtBarrelTo")),
      theMinPtEndcap(iConfig.getParameter<double>("SetMinPtEndcapTo")),
      theMinPL1Tk(iConfig.getParameter<double>("MinPL1Tk")),
      theMinPtL1TkBarrel(iConfig.getParameter<double>("MinPtL1TkBarrel")),
      useOfflineSeed(iConfig.getUntrackedParameter<bool>("UseOfflineSeed", false)),
      useUnassociatedL1(iConfig.getParameter<bool>("UseUnassociatedL1")),
      matchingDR(iConfig.getParameter<std::vector<double>>("MatchDR")),
      etaBins(iConfig.getParameter<std::vector<double>>("EtaMatchingBins"))
{
  muCollToken_ = consumes<l1t::TkMuonCollection>(theSource);

  if (useOfflineSeed) {
    theOfflineSeedLabel = iConfig.getUntrackedParameter<InputTag>("OfflineSeedLabel");
    offlineSeedToken_ = consumes<edm::View<TrajectorySeed>>(theOfflineSeedLabel);

    // check that number of eta bins -1 matches number of dR cones
    if (matchingDR.size() != etaBins.size() - 1) {
      throw cms::Exception("Configuration") << "Size of MatchDR "
                                            << "does not match number of eta bins." << endl;
    }
  }

  // service parameters
  ParameterSet serviceParameters = iConfig.getParameter<ParameterSet>("ServiceParameters");

  // the services
  theService = new MuonServiceProxy(serviceParameters, consumesCollector());

  // the estimator
  theEstimator = new Chi2MeasurementEstimator(10000.);

  produces<L2MuonTrajectorySeedCollection>();
}

// destructor
L2MuonSeedGeneratorFromL1TkMu::~L2MuonSeedGeneratorFromL1TkMu() {
  if (theService)
    delete theService;
  if (theEstimator)
    delete theEstimator;
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

  edm::Handle<l1t::TkMuonCollection> muColl;
  iEvent.getByToken(muCollToken_, muColl);
  LogDebug(metname) << "Number of muons " << muColl->size() << endl;

  edm::Handle<edm::View<TrajectorySeed>> offlineSeedHandle;
  vector<int> offlineSeedMap;
  if (useOfflineSeed) {
    iEvent.getByToken(offlineSeedToken_, offlineSeedHandle);
    offlineSeedMap = vector<int>(offlineSeedHandle->size(), 0);
  }

  for (auto ittkmu = muColl->begin(); ittkmu != muColl->end(); ittkmu++) {

    // L1 tracker track
    auto it = ittkmu->trkPtr();

    // propagate to GMT
    auto p3 = it->momentum();
    float tk_pt = p3.perp();
    float tk_p = p3.mag();
    float tk_eta = p3.eta();
    float tk_aeta = std::abs(tk_eta);
    float tk_phi = p3.phi();
    float tk_q = it->rInv() > 0 ? 1. : -1.;
    float tk_z = it->POCA().z();

    if (tk_p < theMinPL1Tk)
      continue;
    if (tk_aeta < 1.1 && tk_pt < theMinPtL1TkBarrel)
      continue;

    float dzCorrPhi = 1.;
    float deta = 0;
    float etaProp = tk_aeta;

    if (tk_aeta < 1.1) {
      etaProp = 1.1;
      deta = tk_z / 550. / cosh(tk_aeta);
    } else {
      float delta = tk_z / 850.;  //roughly scales as distance to 2nd station
      if (tk_eta > 0)
        delta *= -1;
      dzCorrPhi = 1. + delta;

      float zOzs = tk_z / 850.;
      if (tk_eta > 0)
        deta = zOzs / (1. - zOzs);
      else
        deta = zOzs / (1. + zOzs);
      deta = deta * tanh(tk_eta);
    }
    float resPhi = tk_phi - 1.464 * tk_q * cosh(1.7) / cosh(etaProp) / tk_pt * dzCorrPhi - M_PI / 144.;
    resPhi = reco::reduceRange(resPhi);

    float pt = tk_pt;  //not corrected for eloss
    float eta = tk_eta + deta;
    float theta = 2 * atan(exp(-eta));
    float phi = resPhi;
    int charge = it->rInv() > 0 ? 1 : -1;

    bool barrel = tk_aeta < 1.1 ? true : false;

    if (pt < theL1MinPt || fabs(eta) > theL1MaxEta)
      continue;

    LogDebug(metname) << "New L2 Muon Seed";
    LogDebug(metname) << "Pt = " << pt << " GeV/c";
    LogDebug(metname) << "eta = " << eta;
    LogDebug(metname) << "theta = " << theta << " rad";
    LogDebug(metname) << "phi = " << phi << " rad";
    LogDebug(metname) << "charge = " << charge;
    LogDebug(metname) << "In Barrel? = " << barrel;

    // Update the services
    theService->update(iSetup);

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
      detLayer = theService->detLayerGeometry()->idToLayer(theid);
      LogDebug(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

      const BoundSurface *sur = &(detLayer->surface());
      const BoundCylinder *bc = dynamic_cast<const BoundCylinder *>(sur);

      radius = fabs(bc->radius() / sin(theta));

      LogDebug(metname) << "radius " << radius;

      if (pt < theMinPtBarrel)
        pt = theMinPtBarrel;
    } else {
      LogDebug(metname) << "The seed is in the endcap";

      // ME2
      theid = theta < Geom::pi() / 2. ? CSCDetId(1, 2, 0, 0, 0) : CSCDetId(2, 2, 0, 0, 0);

      detLayer = theService->detLayerGeometry()->idToLayer(theid);
      LogDebug(metname) << "L2 Layer: " << debug.dumpLayer(detLayer);

      radius = fabs(detLayer->position().z() / cos(theta));

      if (pt < theMinPtEndcap)
        pt = theMinPtEndcap;
    }

    vec.setMag(radius);

    GlobalPoint pos(vec.x(), vec.y(), vec.z());

    GlobalVector mom(pt * cos(phi), pt * sin(phi), pt * cos(theta) / sin(theta));

    GlobalTrajectoryParameters param(pos, mom, charge, &*theService->magneticField());
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
    TrajectoryStateOnSurface tsos =
        theService->propagator(thePropagatorName)->propagate(state, detLayer->surface());

    LogDebug(metname) << "State after the propagation on the layer";
    LogDebug(metname) << debug.dumpLayer(detLayer);
    LogDebug(metname) << debug.dumpTSOS(tsos);

    double dRcone = matchingDR[0];
    if (fabs(eta) < etaBins.back()) {
      std::vector<double>::iterator lowEdge = std::upper_bound(etaBins.begin(), etaBins.end(), fabs(eta));
      dRcone = matchingDR.at(lowEdge - etaBins.begin() - 1);
    }

    if (tsos.isValid()) {
      edm::OwnVector<TrackingRecHit> container;

      if (useOfflineSeed) {
        // Get the compatible dets on the layer
        std::vector<pair<const GeomDet *, TrajectoryStateOnSurface>> detsWithStates =
            detLayer->compatibleDets(tsos, *theService->propagator(thePropagatorName), *theEstimator);

        if (detsWithStates.empty() && barrel) {
          // Fallback solution using ME2, try again to propagate but using ME2 as reference
          DetId fallback_id;
          theta < Geom::pi() / 2. ? fallback_id = CSCDetId(1, 2, 0, 0, 0) : fallback_id = CSCDetId(2, 2, 0, 0, 0);
          const DetLayer *ME2DetLayer = theService->detLayerGeometry()->idToLayer(fallback_id);

          tsos = theService->propagator(thePropagatorName)->propagate(state, ME2DetLayer->surface());
          detsWithStates =
              ME2DetLayer->compatibleDets(tsos, *theService->propagator(thePropagatorName), *theEstimator);
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
              TrajectorySeed::const_iterator tsci = assoOffseed->recHits().first,
                                             tscie = assoOffseed->recHits().second;
              for (; tsci != tscie; ++tsci) {
                container.push_back(*tsci);
              }
              auto dummyRef = edm::Ref<MuonBxCollection>();
              output->push_back(
                  L2MuonTrajectorySeed(seedTSOS,
                                       container,
                                       alongMomentum,
                                       dummyRef));
            } else {
              if (useUnassociatedL1) {
                // convert the TSOS into a PTSOD
                PTrajectoryStateOnDet const &seedTSOS =
                    trajectoryStateTransform::persistentState(newTSOS, newTSOSDet->geographicalId().rawId());
                auto dummyRef = edm::Ref<MuonBxCollection>();
                output->push_back(
                    L2MuonTrajectorySeed(seedTSOS,
                                         container,
                                         alongMomentum,
                                         dummyRef));
              }
            }
          }
        }
      } else {
        // convert the TSOS into a PTSOD
        PTrajectoryStateOnDet const &seedTSOS = trajectoryStateTransform::persistentState(tsos, theid.rawId());
        auto dummyRef = edm::Ref<MuonBxCollection>();
        output->push_back(
            L2MuonTrajectorySeed(seedTSOS,
                                 container,
                                 alongMomentum,
                                 dummyRef));
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
  const std::string metlabel = "Muon|RecoMuon|L2MuonSeedGeneratorFromL1TkMu";
  MuonPatternRecoDumper debugtmp;

  edm::View<TrajectorySeed>::const_iterator offseed, endOffseed = offseeds->end();
  const TrajectorySeed *selOffseed = nullptr;
  double bestDr = 99999.;
  unsigned int nOffseed(0);
  int lastOffseed(-1);

  for (offseed = offseeds->begin(); offseed != endOffseed; ++offseed, ++nOffseed) {
    if (offseedMap[nOffseed] != 0)
      continue;
    GlobalPoint glbPos = theService->trackingGeometry()
                             ->idToDet(offseed->startingState().detId())
                             ->surface()
                             .toGlobal(offseed->startingState().parameters().position());
    GlobalVector glbMom = theService->trackingGeometry()
                              ->idToDet(offseed->startingState().detId())
                              ->surface()
                              .toGlobal(offseed->startingState().parameters().momentum());

    // Preliminary check
    double preDr = deltaR(newTsos.globalPosition().eta(), newTsos.globalPosition().phi(), glbPos.eta(), glbPos.phi());
    if (preDr > 1.0)
      continue;

    const FreeTrajectoryState offseedFTS(
        glbPos, glbMom, offseed->startingState().parameters().charge(), &*theService->magneticField());
    TrajectoryStateOnSurface offseedTsos =
        theService->propagator(thePropagatorName)->propagate(offseedFTS, newTsos.surface());
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
      double newDr = deltaR(newTsos.globalPosition().eta(),
                            newTsos.globalPosition().phi(),
                            offseedTsos.globalPosition().eta(),
                            offseedTsos.globalPosition().phi());
      LogDebug(metlabel) << "   -- DR = " << newDr << std::endl;
      if (newDr < dRcone && newDr < bestDr) {
        LogDebug(metlabel) << "          --> OK! " << newDr << std::endl << std::endl;
        selOffseed = &*offseed;
        bestDr = newDr;
        offseedMap[nOffseed] = 1;
        if (lastOffseed > -1)
          offseedMap[lastOffseed] = 0;
        lastOffseed = nOffseed;
      } else {
        LogDebug(metlabel) << "          --> Rejected. " << newDr << std::endl << std::endl;
      }
    } else {
      LogDebug(metlabel) << "Invalid offline seed TSOS after propagation!" << std::endl << std::endl;
    }
  }

  return selOffseed;
}


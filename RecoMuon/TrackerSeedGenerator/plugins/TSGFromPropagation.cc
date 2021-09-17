#include "RecoMuon/TrackerSeedGenerator/plugins/TSGFromPropagation.h"

/** \class TSGFromPropagation
 *
 *  \author Chang Liu - Purdue University 
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC)
    : TSGFromPropagation(iConfig, iC, nullptr) {}

TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet& iConfig,
                                       edm::ConsumesCollector& iC,
                                       const MuonServiceProxy* service)
    : theCategory("Muon|RecoMuon|TSGFromPropagation"),
      theMeasTrackerName(iConfig.getParameter<std::string>("MeasurementTrackerName")),
      theService(service),
      theMaxChi2(iConfig.getParameter<double>("MaxChi2")),
      theFixedErrorRescaling(iConfig.getParameter<double>("ErrorRescaling")),
      theUseVertexStateFlag(iConfig.getParameter<bool>("UseVertexState")),
      theUpdateStateFlag(iConfig.getParameter<bool>("UpdateState")),
      theResetMethod([](const edm::ParameterSet& iConfig) {
        auto resetMethod = iConfig.getParameter<std::string>("ResetMethod");
        if (resetMethod != "discrete" && resetMethod != "fixed" && resetMethod != "matrix") {
          edm::LogError("TSGFromPropagation") << "Wrong error rescaling method: " << resetMethod << "\n"
                                              << "Possible choices are: discrete, fixed, matrix.\n"
                                              << "Use discrete method" << std::endl;
          resetMethod = "discrete";
        }
        if ("fixed" == resetMethod) {
          return ResetMethod::fixed;
        }
        if ("matrix" == resetMethod) {
          return ResetMethod::matrix;
        }
        return ResetMethod::discrete;
      }(iConfig)),
      theSelectStateFlag(iConfig.getParameter<bool>("SelectState")),
      thePropagatorName(iConfig.getParameter<std::string>("Propagator")),
      theSigmaZ(iConfig.getParameter<double>("SigmaZ")),
      theErrorMatrixPset(iConfig.getParameter<edm::ParameterSet>("errorMatrixPset")),
      theBeamSpotToken(iC.consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      theMeasurementTrackerEventToken(
          iC.consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))) {}

TSGFromPropagation::~TSGFromPropagation() { LogTrace(theCategory) << " TSGFromPropagation dtor called "; }

void TSGFromPropagation::trackerSeeds(const TrackCand& staMuon,
                                      const TrackingRegion& region,
                                      const TrackerTopology* tTopo,
                                      std::vector<TrajectorySeed>& result) {
  if (theResetMethod == ResetMethod::discrete)
    getRescalingFactor(staMuon);

  TrajectoryStateOnSurface staState = outerTkState(staMuon);

  if (!staState.isValid()) {
    LogTrace(theCategory) << "Error: initial state from L2 muon is invalid.";
    return;
  }

  LogTrace(theCategory) << "begin of trackerSeed:\n staState pos: " << staState.globalPosition()
                        << " mom: " << staState.globalMomentum() << "pos eta: " << staState.globalPosition().eta()
                        << "mom eta: " << staState.globalMomentum().eta();

  std::vector<const DetLayer*> nls = theNavigation->compatibleLayers(*(staState.freeState()), oppositeToMomentum);

  LogTrace(theCategory) << " compatible layers: " << nls.size();

  if (nls.empty())
    return;

  int ndesLayer = 0;

  bool usePredictedState = false;

  if (theUpdateStateFlag) {  //use updated states
    std::vector<TrajectoryMeasurement> alltm;

    for (std::vector<const DetLayer*>::const_iterator inl = nls.begin(); inl != nls.end(); inl++, ndesLayer++) {
      if ((*inl == nullptr))
        break;
      //         if ( (inl != nls.end()-1 ) && ( (*inl)->subDetector() == GeomDetEnumerators::TEC ) && ( (*(inl+1))->subDetector() == GeomDetEnumerators::TOB ) ) continue;
      alltm = findMeasurements(*inl, staState);
      if ((!alltm.empty())) {
        LogTrace(theCategory) << "final compatible layer: " << ndesLayer;
        break;
      }
    }

    if (alltm.empty()) {
      LogTrace(theCategory) << " NO Measurements Found: eta: " << staState.globalPosition().eta() << "pt "
                            << staState.globalMomentum().perp();
      usePredictedState = true;
    } else {
      LogTrace(theCategory) << " Measurements for seeds: " << alltm.size();
      std::stable_sort(alltm.begin(), alltm.end(), increasingEstimate());
      if (alltm.size() > 5)
        alltm.erase(alltm.begin() + 5, alltm.end());

      int i = 0;
      for (std::vector<TrajectoryMeasurement>::const_iterator itm = alltm.begin(); itm != alltm.end(); itm++, i++) {
        TrajectoryStateOnSurface updatedTSOS = updator()->update(itm->predictedState(), *(itm->recHit()));
        if (updatedTSOS.isValid() && passSelection(updatedTSOS)) {
          edm::OwnVector<TrackingRecHit> container;
          container.push_back(itm->recHit()->hit()->clone());
          TrajectorySeed ts = createSeed(updatedTSOS, container, itm->recHit()->geographicalId());
          result.push_back(ts);
        }
      }
      LogTrace(theCategory) << "result: " << result.size();
      return;
    }
  }

  if (!theUpdateStateFlag || usePredictedState) {  //use predicted states
    LogTrace(theCategory) << "use predicted state: ";
    for (std::vector<const DetLayer*>::const_iterator inl = nls.begin(); inl != nls.end(); inl++) {
      if (!result.empty() || *inl == nullptr) {
        break;
      }
      std::vector<DetLayer::DetWithState> compatDets = (*inl)->compatibleDets(staState, *propagator(), *estimator());
      LogTrace(theCategory) << " compatDets " << compatDets.size();
      if (compatDets.empty())
        continue;
      TrajectorySeed ts = createSeed(compatDets.front().second, compatDets.front().first->geographicalId());
      result.push_back(ts);
    }
    LogTrace(theCategory) << "result: " << result.size();
    return;
  }
  return;
}

void TSGFromPropagation::init(const MuonServiceProxy* service) {
  theFlexErrorRescaling = 1.0;

  theEstimator = std::make_unique<Chi2MeasurementEstimator>(theMaxChi2);

  theCacheId_MT = 0;

  theCacheId_TG = 0;

  theService = service;

  theUpdator = std::make_unique<KFUpdator>();

  if (theResetMethod == ResetMethod::matrix && !theErrorMatrixPset.empty()) {
    theErrorMatrixAdjuster = std::make_unique<MuonErrorMatrix>(theErrorMatrixPset);
  }

  theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theTracker);
  theNavigation = std::make_unique<DirectTrackerNavigation>(theTracker);
}

void TSGFromPropagation::setEvent(const edm::Event& iEvent) {
  iEvent.getByToken(theBeamSpotToken, beamSpot);

  unsigned long long newCacheId_MT = theService->eventSetup().get<CkfComponentsRecord>().cacheIdentifier();

  if (theUpdateStateFlag && newCacheId_MT != theCacheId_MT) {
    LogTrace(theCategory) << "Measurment Tracker Geometry changed!";
    theCacheId_MT = newCacheId_MT;
    theService->eventSetup().get<CkfComponentsRecord>().get(theMeasTrackerName, theMeasTracker);
  }

  if (theUpdateStateFlag) {
    iEvent.getByToken(theMeasurementTrackerEventToken, theMeasTrackerEvent);
  }

  bool trackerGeomChanged = false;

  unsigned long long newCacheId_TG = theService->eventSetup().get<TrackerRecoGeometryRecord>().cacheIdentifier();

  if (newCacheId_TG != theCacheId_TG) {
    LogTrace(theCategory) << "Tracker Reco Geometry changed!";
    theCacheId_TG = newCacheId_TG;
    theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theTracker);
    trackerGeomChanged = true;
  }

  if (trackerGeomChanged && (theTracker.product() != nullptr)) {
    theNavigation = std::make_unique<DirectTrackerNavigation>(theTracker);
  }
}

TrajectoryStateOnSurface TSGFromPropagation::innerState(const TrackCand& staMuon) const {
  TrajectoryStateOnSurface innerTS;

  if (staMuon.first && staMuon.first->isValid()) {
    if (staMuon.first->direction() == alongMomentum) {
      innerTS = staMuon.first->firstMeasurement().updatedState();
    } else if (staMuon.first->direction() == oppositeToMomentum) {
      innerTS = staMuon.first->lastMeasurement().updatedState();
    }
  } else {
    innerTS = trajectoryStateTransform::innerStateOnSurface(
        *(staMuon.second), *theService->trackingGeometry(), &*theService->magneticField());
  }
  //rescale the error
  adjust(innerTS);

  return innerTS;

  //    return trajectoryStateTransform::innerStateOnSurface(*(staMuon.second),*theService->trackingGeometry(), &*theService->magneticField());
}

TrajectoryStateOnSurface TSGFromPropagation::outerTkState(const TrackCand& staMuon) const {
  TrajectoryStateOnSurface result;

  if (theUseVertexStateFlag && staMuon.second->pt() > 1.0) {
    FreeTrajectoryState iniState =
        trajectoryStateTransform::initialFreeState(*(staMuon.second), &*theService->magneticField());
    //rescale the error at IP
    adjust(iniState);

    StateOnTrackerBound fromInside(&*(theService->propagator("PropagatorWithMaterial")));
    result = fromInside(iniState);
  } else {
    StateOnTrackerBound fromOutside(&*propagator());
    result = fromOutside(innerState(staMuon));
  }
  return result;
}

TrajectorySeed TSGFromPropagation::createSeed(const TrajectoryStateOnSurface& tsos, const DetId& id) const {
  edm::OwnVector<TrackingRecHit> container;
  return createSeed(tsos, container, id);
}

TrajectorySeed TSGFromPropagation::createSeed(const TrajectoryStateOnSurface& tsos,
                                              const edm::OwnVector<TrackingRecHit>& container,
                                              const DetId& id) const {
  PTrajectoryStateOnDet const& seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());
  return TrajectorySeed(seedTSOS, container, oppositeToMomentum);
}

void TSGFromPropagation::validMeasurements(std::vector<TrajectoryMeasurement>& tms) const {
  std::vector<TrajectoryMeasurement>::iterator tmsend = std::remove_if(tms.begin(), tms.end(), isInvalid());
  tms.erase(tmsend, tms.end());
  return;
}

std::vector<TrajectoryMeasurement> TSGFromPropagation::findMeasurements(
    const DetLayer* nl, const TrajectoryStateOnSurface& staState) const {
  std::vector<TrajectoryMeasurement> result;

  std::vector<DetLayer::DetWithState> compatDets = nl->compatibleDets(staState, *propagator(), *estimator());
  if (compatDets.empty())
    return result;

  for (std::vector<DetLayer::DetWithState>::const_iterator idws = compatDets.begin(); idws != compatDets.end();
       ++idws) {
    if (idws->second.isValid() && (idws->first)) {
      std::vector<TrajectoryMeasurement> tmptm =
          theMeasTrackerEvent->idToDet(idws->first->geographicalId())
              .fastMeasurements(idws->second, idws->second, *propagator(), *estimator());
      validMeasurements(tmptm);
      //         if ( tmptm.size() > 2 ) {
      //            std::stable_sort(tmptm.begin(),tmptm.end(),increasingEstimate());
      //            result.insert(result.end(),tmptm.begin(), tmptm.begin()+2);
      //         } else {
      result.insert(result.end(), tmptm.begin(), tmptm.end());
      //         }
    }
  }

  return result;
}

bool TSGFromPropagation::passSelection(const TrajectoryStateOnSurface& tsos) const {
  if (!theSelectStateFlag)
    return true;
  else {
    if (beamSpot.isValid()) {
      return ((fabs(zDis(tsos) - beamSpot->z0()) < theSigmaZ));

    } else {
      return ((fabs(zDis(tsos)) < theSigmaZ));
      //      double theDxyCut = 100;
      //      return ( (zDis(tsos) < theSigmaZ) && (dxyDis(tsos) < theDxyCut) );
    }
  }
}

double TSGFromPropagation::dxyDis(const TrajectoryStateOnSurface& tsos) const {
  return fabs(
      (-tsos.globalPosition().x() * tsos.globalMomentum().y() + tsos.globalPosition().y() * tsos.globalMomentum().x()) /
      tsos.globalMomentum().perp());
}

double TSGFromPropagation::zDis(const TrajectoryStateOnSurface& tsos) const {
  return tsos.globalPosition().z() -
         tsos.globalPosition().perp() * tsos.globalMomentum().z() / tsos.globalMomentum().perp();
}

void TSGFromPropagation::getRescalingFactor(const TrackCand& staMuon) {
  float pt = (staMuon.second)->pt();
  if (pt < 13.0)
    theFlexErrorRescaling = 3;
  else if (pt < 30.0)
    theFlexErrorRescaling = 5;
  else
    theFlexErrorRescaling = 10;
  return;
}

void TSGFromPropagation::adjust(FreeTrajectoryState& state) const {
  //rescale the error
  if (theResetMethod == ResetMethod::discrete) {
    state.rescaleError(theFlexErrorRescaling);
    return;
  }

  //rescale the error
  if (theResetMethod == ResetMethod::fixed || !theErrorMatrixAdjuster) {
    state.rescaleError(theFixedErrorRescaling);
    return;
  }

  CurvilinearTrajectoryError oMat = state.curvilinearError();
  CurvilinearTrajectoryError sfMat = theErrorMatrixAdjuster->get(state.momentum());  //FIXME with position
  MuonErrorMatrix::multiply(oMat, sfMat);

  state = FreeTrajectoryState(state.parameters(), oMat);
}

void TSGFromPropagation::adjust(TrajectoryStateOnSurface& state) const {
  //rescale the error
  if (theResetMethod == ResetMethod::discrete) {
    state.rescaleError(theFlexErrorRescaling);
    return;
  }

  if (theResetMethod == ResetMethod::fixed || !theErrorMatrixAdjuster) {
    state.rescaleError(theFixedErrorRescaling);
    return;
  }

  CurvilinearTrajectoryError oMat = state.curvilinearError();
  CurvilinearTrajectoryError sfMat = theErrorMatrixAdjuster->get(state.globalMomentum());  //FIXME with position
  MuonErrorMatrix::multiply(oMat, sfMat);

  state =
      TrajectoryStateOnSurface(state.weight(), state.globalParameters(), oMat, state.surface(), state.surfaceSide());
}

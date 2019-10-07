#include "FastSimulation/Muons/plugins/FastTSGFromPropagation.h"

/** \class FastTSGFromPropagation
 *
 *  Emulate TSGFromPropagation in RecoMuon
 *
 *  \author Hwidong Yoo - Purdue University 
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
// #include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "FastSimulation/Tracking/interface/TrajectorySeedHitCandidate.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FastSimulation/Tracking/interface/FastTrackingUtilities.h"

using namespace std;

FastTSGFromPropagation::FastTSGFromPropagation(const edm::ParameterSet& iConfig, edm::ConsumesCollector& iC)
    : FastTSGFromPropagation(iConfig, nullptr, iC) {}

FastTSGFromPropagation::FastTSGFromPropagation(const edm::ParameterSet& iConfig,
                                               const MuonServiceProxy* service,
                                               edm::ConsumesCollector& iC)
    : theCategory("FastSimulation|Muons|FastTSGFromPropagation"),
      theTracker(),
      theNavigation(),
      theService(service),
      theUpdator(),
      theEstimator(),
      theSigmaZ(0.0),
      theConfig(iConfig),
      theSimTrackCollectionToken_(
          iC.consumes<edm::SimTrackContainer>(theConfig.getParameter<edm::InputTag>("SimTrackCollectionLabel"))),
      recHitCombinationsToken_(
          iC.consumes<FastTrackerRecHitCombinationCollection>(theConfig.getParameter<edm::InputTag>("HitProducer"))),
      beamSpot_(iC.consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
      theMeasurementTrackerEventToken_(
          iC.consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("MeasurementTrackerEvent"))) {}

FastTSGFromPropagation::~FastTSGFromPropagation() { LogTrace(theCategory) << " FastTSGFromPropagation dtor called "; }

void FastTSGFromPropagation::trackerSeeds(const TrackCand& staMuon,
                                          const TrackingRegion& region,
                                          const TrackerTopology* tTopo,
                                          std::vector<TrajectorySeed>& result) {
  if (theResetMethod == "discrete")
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

      const edm::SimTrackContainer* simTracks = &(*theSimTracks);
      TrajectorySeedHitCandidate theSeedHits;
      std::vector<TrajectorySeedHitCandidate> outerHits;

      //std::vector<TrajectorySeed>  tmpTS;
      bool isMatch = false;
      for (std::vector<TrajectoryMeasurement>::const_iterator itm = alltm.begin(); itm != alltm.end(); itm++) {
        const TrajectoryStateOnSurface seedState = itm->predictedState();
        double preY = seedState.globalPosition().y();

        // Check SimTrack
        FreeTrajectoryState simtrack_trackerstate;
        for (unsigned icomb = 0; icomb < recHitCombinations->size(); ++icomb) {
          const auto& recHitCombination = (*recHitCombinations)[icomb];
          if (recHitCombination.empty())
            continue;
          int32_t simTrackId = recHitCombination.back()->simTrackId(0);
          const SimTrack& simtrack = (*simTracks)[simTrackId];

          GlobalPoint position(simtrack.trackerSurfacePosition().x(),
                               simtrack.trackerSurfacePosition().y(),
                               simtrack.trackerSurfacePosition().z());
          GlobalVector momentum(simtrack.trackerSurfaceMomentum().x(),
                                simtrack.trackerSurfaceMomentum().y(),
                                simtrack.trackerSurfaceMomentum().z());
          int charge = (int)simtrack.charge();
          GlobalTrajectoryParameters glb_parameters(
              position, momentum, charge, &*theService->magneticField().product());
          simtrack_trackerstate = FreeTrajectoryState(glb_parameters);

          unsigned int outerId = 0;
          for (const auto& recHitRef : recHitCombination) {
            theSeedHits = TrajectorySeedHitCandidate(recHitRef.get(), tTopo);
            unsigned int id = theSeedHits.hit()->geographicalId().rawId();
            if (preY < 0) {
              if (id > outerId)
                outerId = id;
            } else {
              if (id > outerId)
                outerId = id;
            }
          }
          for (const auto& recHitRef : recHitCombination) {
            theSeedHits = TrajectorySeedHitCandidate(recHitRef.get(), tTopo);
            if (itm->recHit()->hit()->geographicalId().rawId() == theSeedHits.hit()->geographicalId().rawId()) {
              auto aTrackingRecHit = std::unique_ptr<TrackingRecHit>(theSeedHits.hit()->clone());
              TransientTrackingRecHit::ConstRecHitPointer recHit = theTTRHBuilder->build(aTrackingRecHit.get());
              if (!recHit)
                continue;
              TrajectoryStateOnSurface updatedTSOS = updator()->update(seedState, *(recHit));
              if (updatedTSOS.isValid() && passSelection(updatedTSOS)) {
                edm::OwnVector<TrackingRecHit> container;
                container.push_back(recHit->hit()->clone());
                fastTrackingUtilities::setRecHitCombinationIndex(container, icomb);
                TrajectorySeed ts = createSeed(updatedTSOS, container, recHit->geographicalId());
                // check direction
                const TrajectorySeed* aSeed = &ts;
                PTrajectoryStateOnDet PTSOD = aSeed->startingState();

                const GeomDet* g = theGeometry->idToDet(PTSOD.detId());
                TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(
                    PTSOD, &(g->surface()), &*theService->magneticField().product());
                if (tsos.globalMomentum().basicVector() * seedState.globalMomentum().basicVector() < 0.)
                  continue;
                result.push_back(ts);
                isMatch = true;
              }
            }
          }
        }
      }
      if (!isMatch) {
        // if there is no hits w.r.t. TM, find outermost hit
        for (std::vector<TrajectoryMeasurement>::const_iterator itm = alltm.begin(); itm != alltm.end(); itm++) {
          const TrajectoryStateOnSurface seedState = itm->predictedState();
          double preY = seedState.globalPosition().y();

          // Check SimTrack
          TrackingRecHit* aTrackingRecHit;
          FreeTrajectoryState simtrack_trackerstate;

          for (unsigned icomb = 0; icomb < recHitCombinations->size(); ++icomb) {
            const auto& recHitCombination = (*recHitCombinations)[icomb];
            if (recHitCombination.empty())
              continue;
            int32_t simTrackId = recHitCombination.back()->simTrackId(0);
            const SimTrack& simtrack = (*simTracks)[simTrackId];

            GlobalPoint position(simtrack.trackerSurfacePosition().x(),
                                 simtrack.trackerSurfacePosition().y(),
                                 simtrack.trackerSurfacePosition().z());
            GlobalVector momentum(simtrack.trackerSurfaceMomentum().x(),
                                  simtrack.trackerSurfaceMomentum().y(),
                                  simtrack.trackerSurfaceMomentum().z());
            int charge = (int)simtrack.charge();
            GlobalTrajectoryParameters glb_parameters(
                position, momentum, charge, &*theService->magneticField().product());
            simtrack_trackerstate = FreeTrajectoryState(glb_parameters);

            unsigned int outerId = 0;
            for (const auto& recHitRef : recHitCombination) {
              theSeedHits = TrajectorySeedHitCandidate(recHitRef.get(), tTopo);
              unsigned int id = theSeedHits.hit()->geographicalId().rawId();
              if (preY < 0) {
                if (id > outerId)
                  outerId = id;
              } else {
                if (id > outerId)
                  outerId = id;
              }
            }
            for (const auto& recHitRef : recHitCombination) {
              theSeedHits = TrajectorySeedHitCandidate(recHitRef.get(), tTopo);
              if (outerId == theSeedHits.hit()->geographicalId().rawId()) {
                aTrackingRecHit = theSeedHits.hit()->clone();
                TransientTrackingRecHit::ConstRecHitPointer recHit = theTTRHBuilder->build(aTrackingRecHit);
                if (!recHit)
                  continue;
                TrajectoryStateOnSurface updatedTSOS = updator()->update(seedState, *(recHit));
                if (updatedTSOS.isValid() && passSelection(updatedTSOS)) {
                  edm::OwnVector<TrackingRecHit> container;
                  container.push_back(recHit->hit()->clone());
                  fastTrackingUtilities::setRecHitCombinationIndex(container, icomb);
                  TrajectorySeed ts = createSeed(updatedTSOS, container, recHit->geographicalId());
                  // check direction
                  const TrajectorySeed* aSeed = &ts;
                  PTrajectoryStateOnDet PTSOD = aSeed->startingState();

                  const GeomDet* g = theGeometry->idToDet(PTSOD.detId());
                  TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(
                      PTSOD, &(g->surface()), &*theService->magneticField().product());
                  if (tsos.globalMomentum().basicVector() * seedState.globalMomentum().basicVector() < 0.)
                    continue;
                  result.push_back(ts);
                }
              }
            }
          }
        }
      }

      /*
       for( unsigned ir = 0; ir < tmpTS.size(); ir++ ) {
	 const BasicTrajectorySeed* aSeed = &((tmpTS)[ir]);
	 PTrajectoryStateOnDet PTSOD = aSeed->startingState();
	 
	 DetId seedDetId(PTSOD.detId());
	 const GeomDet * g = theGeometry->idToDet(seedDetId);
	 TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState(PTSOD, &(g->surface()),  &*theService->magneticField().product());
		 cout << "tsos3 = " << tsos.globalMomentum() << endl;
	 if( _index == ir ) {
		 cout << "tsos4 = " << tsos.globalMomentum() << endl;
	     result.push_back(tmpTS[ir]);
	 }
       }
       */
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

void FastTSGFromPropagation::init(const MuonServiceProxy* service) {
  theMaxChi2 = theConfig.getParameter<double>("MaxChi2");

  theFixedErrorRescaling = theConfig.getParameter<double>("ErrorRescaling");

  theFlexErrorRescaling = 1.0;

  theResetMethod = theConfig.getParameter<std::string>("ResetMethod");

  if (theResetMethod != "discrete" && theResetMethod != "fixed" && theResetMethod != "matrix") {
    edm::LogError("FastTSGFromPropagation") << "Wrong error rescaling method: " << theResetMethod << "\n"
                                            << "Possible choices are: discrete, fixed, matrix.\n"
                                            << "Use discrete method" << std::endl;
    theResetMethod = "discrete";
  }

  theEstimator.reset(new Chi2MeasurementEstimator(theMaxChi2));

  theCacheId_MT = 0;

  theCacheId_TG = 0;

  thePropagatorName = theConfig.getParameter<std::string>("Propagator");

  theService = service;

  theUseVertexStateFlag = theConfig.getParameter<bool>("UseVertexState");

  theUpdateStateFlag = theConfig.getParameter<bool>("UpdateState");

  theSelectStateFlag = theConfig.getParameter<bool>("SelectState");

  theUpdator.reset(new KFUpdator());

  theSigmaZ = theConfig.getParameter<double>("SigmaZ");

  edm::ParameterSet errorMatrixPset = theConfig.getParameter<edm::ParameterSet>("errorMatrixPset");
  if (theResetMethod == "matrix" && !errorMatrixPset.empty()) {
    theAdjustAtIp = errorMatrixPset.getParameter<bool>("atIP");
    theErrorMatrixAdjuster.reset(new MuonErrorMatrix(errorMatrixPset));
  } else {
    theAdjustAtIp = false;
    theErrorMatrixAdjuster.reset();
  }

  theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theTracker);
  theNavigation.reset(new DirectTrackerNavigation(theTracker));

  edm::ESHandle<TrackerGeometry> geometry;
  theService->eventSetup().get<TrackerDigiGeometryRecord>().get(geometry);
  theGeometry = &(*geometry);

  theService->eventSetup().get<TransientRecHitRecord>().get("WithTrackAngle", theTTRHBuilder);
}

void FastTSGFromPropagation::setEvent(const edm::Event& iEvent) {
  iEvent.getByToken(beamSpot_, theBeamSpot);

  // retrieve the MC truth (SimTracks)
  iEvent.getByToken(theSimTrackCollectionToken_, theSimTracks);
  iEvent.getByToken(recHitCombinationsToken_, recHitCombinations);

  unsigned long long newCacheId_MT = theService->eventSetup().get<CkfComponentsRecord>().cacheIdentifier();

  if (theUpdateStateFlag && newCacheId_MT != theCacheId_MT) {
    LogTrace(theCategory) << "Measurment Tracker Geometry changed!";
    theCacheId_MT = newCacheId_MT;
    theService->eventSetup().get<CkfComponentsRecord>().get(theMeasTracker);
  }

  if (theUpdateStateFlag) {
    iEvent.getByToken(theMeasurementTrackerEventToken_, theMeasTrackerEvent);
  }

  bool trackerGeomChanged = false;

  unsigned long long newCacheId_TG = theService->eventSetup().get<TrackerRecoGeometryRecord>().cacheIdentifier();

  if (newCacheId_TG != theCacheId_TG) {
    LogTrace(theCategory) << "Tracker Reco Geometry changed!";
    theCacheId_TG = newCacheId_TG;
    theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theTracker);
    trackerGeomChanged = true;
  }

  if (trackerGeomChanged && theTracker.product()) {
    theNavigation.reset(new DirectTrackerNavigation(theTracker));
  }
}

TrajectoryStateOnSurface FastTSGFromPropagation::innerState(const TrackCand& staMuon) const {
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
}

TrajectoryStateOnSurface FastTSGFromPropagation::outerTkState(const TrackCand& staMuon) const {
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

TrajectorySeed FastTSGFromPropagation::createSeed(const TrajectoryStateOnSurface& tsos, const DetId& id) const {
  edm::OwnVector<TrackingRecHit> container;
  return createSeed(tsos, container, id);
}

TrajectorySeed FastTSGFromPropagation::createSeed(const TrajectoryStateOnSurface& tsos,
                                                  const edm::OwnVector<TrackingRecHit>& container,
                                                  const DetId& id) const {
  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());
  return TrajectorySeed(seedTSOS, container, oppositeToMomentum);
}

void FastTSGFromPropagation::validMeasurements(std::vector<TrajectoryMeasurement>& tms) const {
  std::vector<TrajectoryMeasurement>::iterator tmsend = std::remove_if(tms.begin(), tms.end(), isInvalid());
  tms.erase(tmsend, tms.end());
  return;
}

std::vector<TrajectoryMeasurement> FastTSGFromPropagation::findMeasurements(
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
      //validMeasurements(tmptm);
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

bool FastTSGFromPropagation::passSelection(const TrajectoryStateOnSurface& tsos) const {
  if (!theSelectStateFlag)
    return true;
  else {
    if (theBeamSpot.isValid()) {
      return ((fabs(zDis(tsos) - theBeamSpot->z0()) < theSigmaZ));

    } else {
      return ((fabs(zDis(tsos)) < theSigmaZ));
      //      double theDxyCut = 100;
      //      return ( (zDis(tsos) < theSigmaZ) && (dxyDis(tsos) < theDxyCut) );
    }
  }
}

double FastTSGFromPropagation::dxyDis(const TrajectoryStateOnSurface& tsos) const {
  return fabs(
      (-tsos.globalPosition().x() * tsos.globalMomentum().y() + tsos.globalPosition().y() * tsos.globalMomentum().x()) /
      tsos.globalMomentum().perp());
}

double FastTSGFromPropagation::zDis(const TrajectoryStateOnSurface& tsos) const {
  return tsos.globalPosition().z() -
         tsos.globalPosition().perp() * tsos.globalMomentum().z() / tsos.globalMomentum().perp();
}

void FastTSGFromPropagation::getRescalingFactor(const TrackCand& staMuon) {
  float pt = (staMuon.second)->pt();
  if (pt < 13.0)
    theFlexErrorRescaling = 3;
  else if (pt < 30.0)
    theFlexErrorRescaling = 5;
  else
    theFlexErrorRescaling = 10;
  return;
}

void FastTSGFromPropagation::adjust(FreeTrajectoryState& state) const {
  //rescale the error
  if (theResetMethod == "discreate") {
    state.rescaleError(theFlexErrorRescaling);
    return;
  }

  //rescale the error
  if (theResetMethod == "fixed" || !theErrorMatrixAdjuster) {
    state.rescaleError(theFixedErrorRescaling);
    return;
  }

  CurvilinearTrajectoryError oMat = state.curvilinearError();
  CurvilinearTrajectoryError sfMat = theErrorMatrixAdjuster->get(state.momentum());  //FIXME with position
  MuonErrorMatrix::multiply(oMat, sfMat);

  state = FreeTrajectoryState(state.parameters(), oMat);
}

void FastTSGFromPropagation::adjust(TrajectoryStateOnSurface& state) const {
  //rescale the error
  if (theResetMethod == "discreate") {
    state.rescaleError(theFlexErrorRescaling);
    return;
  }

  if (theResetMethod == "fixed" || !theErrorMatrixAdjuster) {
    state.rescaleError(theFixedErrorRescaling);
    return;
  }

  CurvilinearTrajectoryError oMat = state.curvilinearError();
  CurvilinearTrajectoryError sfMat = theErrorMatrixAdjuster->get(state.globalMomentum());  //FIXME with position
  MuonErrorMatrix::multiply(oMat, sfMat);

  state =
      TrajectoryStateOnSurface(state.weight(), state.globalParameters(), oMat, state.surface(), state.surfaceSide());
}

void FastTSGFromPropagation::stateOnDet(const TrajectoryStateOnSurface& ts,
                                        unsigned int detid,
                                        PTrajectoryStateOnDet& pts) const {
  const AlgebraicSymMatrix55& m = ts.localError().matrix();
  int dim = 5;  /// should check if corresponds to m
  float localErrors[15];
  int k = 0;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j <= i; ++j) {
      localErrors[k++] = m(i, j);
    }
  }
  int surfaceSide = static_cast<int>(ts.surfaceSide());
  pts = PTrajectoryStateOnDet(ts.localParameters(), ts.globalMomentum().perp(), localErrors, detid, surfaceSide);
}

#include "RecoMuon/TrackerSeedGenerator/plugins/TSGFromPropagation.h"

/** \class TSGFromPropagation
 *
 *  $Date: 2008/05/08 15:00:19 $
 *  $Revision: 1.25 $
 *  \author Chang Liu - Purdue University 
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoMuon/GlobalTrackingTools/interface/DirectTrackerNavigation.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"


TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet & iConfig) :theTkLayerMeasurements (0), theTracker(0), theMeasTracker(0), theNavigation(0), theService(0), theEstimator(0), theTSTransformer(0), theConfig (iConfig)
{
  theCategory = "Muon|RecoMuon|TSGFromPropagation";
}

TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet & iConfig, const MuonServiceProxy* service) : theTkLayerMeasurements (0), theTracker(0), theMeasTracker(0), theNavigation(0), theService(service),theUpdator(0), theEstimator(0), theTSTransformer(0), theConfig (iConfig)
{
}


TSGFromPropagation::~TSGFromPropagation()
{

  LogTrace(theCategory) << " TSGFromPropagation dtor called ";

  if ( theNavigation ) delete theNavigation;
  if ( theUpdator ) delete theUpdator;
  if ( theEstimator ) delete theEstimator;
  if ( theTkLayerMeasurements ) delete theTkLayerMeasurements;
  if ( theTSTransformer ) delete  theTSTransformer;
  if ( theErrorMatrixAdjuster ) delete theErrorMatrixAdjuster;
  LogTrace(theCategory) << " TSGFromPropagation dtor finished  ";

}

void TSGFromPropagation::trackerSeeds(const TrackCand& staMuon, const TrackingRegion& region, std::vector<TrajectorySeed> & result) {

  LogTrace(theCategory) << " begin of trackerSeed ";

  TrajectoryStateOnSurface staState = outerTkState(staMuon);

  if ( !staState.isValid() ) { 
    LogTrace(theCategory) << " initial state invalid, fail";
    return;
  }

  LogTrace(theCategory) << " staState pos: "<<staState.globalPosition()
                        << " mom: "<<staState.globalMomentum() 
                        <<"pos eta: "<<staState.globalPosition().eta()
                        <<"mom eta: "<<staState.globalMomentum().eta();

  std::vector<const DetLayer*> nls = theNavigation->compatibleLayers(*(staState.freeState()), oppositeToMomentum);

  LogTrace(theCategory) << " compatible layers: "<<nls.size();

  if ( nls.empty() ) return;

  int ndesLayer = 0;

  bool usePredictedState = false;

  if ( theUpdateStateFlag ) { //use updated states
     std::vector<TrajectoryMeasurement> alltm; 

     for (std::vector<const DetLayer*>::const_iterator inl = nls.begin();
         inl != nls.end(); inl++ ) {

         if ( (!alltm.empty()) || (*inl == 0) ) {
            LogTrace(theCategory) << "final compatible layer: "<<ndesLayer;
            if (debug_) h_NseedLayer->Fill(ndesLayer);
            break;
         }
         ndesLayer++;
         std::vector<TrajectoryMeasurement> tmptm = findMeasurements_new(*inl, staState);
         LogTrace(theCategory) << " Number of measurements in used layer: "<<alltm.size();
         if ( tmptm.empty() )  continue;
         alltm.insert(alltm.end(),tmptm.begin(), tmptm.end());
     }

     if ( alltm.empty() ) {
        LogTrace(theCategory) << " NO Measurements Found: eta: "<<staState.globalPosition().eta() <<"pt "<<staState.globalMomentum().perp();
        if (debug_) h_Eta_updatingFail->Fill(staState.globalPosition().eta());
        if (debug_) h_Pt_updatingFail->Fill(staState.globalMomentum().perp());
        usePredictedState = true;
     } else {
       LogTrace(theCategory) << " Measurements for seeds: "<<alltm.size();
       if (debug_) h_NrawMeas->Fill(alltm.size());
       selectMeasurements(alltm);
       LogTrace(theCategory) << " Measurements for seeds after select: "<<alltm.size();

       for (std::vector<TrajectoryMeasurement>::const_iterator itm = alltm.begin();
            itm != alltm.end(); itm++) {
            if (debug_) h_chi2->Fill(itm->estimate());
            TrajectoryStateOnSurface updatedTSOS = updator()->update(itm->predictedState(), *(itm->recHit()));
            if ( updatedTSOS.isValid() )  {
               edm::OwnVector<TrackingRecHit> container;
               container.push_back(itm->recHit()->hit()->clone());
               TrajectorySeed ts = createSeed(updatedTSOS, container, itm->recHit()->geographicalId());
               result.push_back(ts);
            }
       }
     if (debug_) h_NupdatedSeeds->Fill(result.size());
     return;
    }
  }

  if ( !theUpdateStateFlag || usePredictedState ) { //use predicted states
     LogTrace(theCategory) << "use predicted state: ";
     for (std::vector<const DetLayer*>::const_iterator inl = nls.begin();
         inl != nls.end(); inl++ ) {

         if ( !result.empty() || *inl == 0 ) {
            break;
         }
         std::vector<DetLayer::DetWithState> compatDets = (*inl)->compatibleDets(staState, *propagator(), *estimator());
         LogTrace(theCategory) << " compatDets "<<compatDets.size();
         if ( compatDets.empty() ) continue;
         TrajectorySeed ts = createSeed(compatDets.front().second, compatDets.front().first->geographicalId());
         result.push_back(ts);

     }
     LogTrace(theCategory) << "result: "<<result.size();
     return;
  } 
  return;
}
    

void TSGFromPropagation::init(const MuonServiceProxy* service) {

  theMaxChi2 = theConfig.getParameter<double>("MaxChi2");

  theErrorRescaling = theConfig.getParameter<double>("ErrorRescaling");

  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);

  theCacheId_MT = 0;

  thePropagatorName = theConfig.getParameter<std::string>("Propagator");

  theService = service;

  theUseVertexStateFlag = theConfig.getParameter<bool>("UseVertexState");

  theUpdateStateFlag = theConfig.getParameter<bool>("UpdateState");

  theUseSecondMeasurementsFlag = theConfig.getParameter<bool>("UseSecondMeasurements");

  theUpdator = new KFUpdator();

  theTSTransformer = new TrajectoryStateTransform();

  edm::ParameterSet errorMatrixPset = theConfig.getParameter<edm::ParameterSet>("errorMatrixPset");
  if (!errorMatrixPset.empty()){
    theAdjustAtIp = errorMatrixPset.getParameter<bool>("atIP");
    theErrorMatrixAdjuster = new MuonErrorMatrix(errorMatrixPset);
  } else {
    theAdjustAtIp =false;
    theErrorMatrixAdjuster=0;
  }

  theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theTracker); 
  theNavigation = new DirectTrackerNavigation(theTracker);

  debug_ = theConfig.getUntrackedParameter<bool>("Debug",false);
  if ( debug_ ) {
    edm::Service<TFileService> fs;
    h_chi2 = fs->make<TH1F>("h_chi2","h_chi2",30,0,30);
    h_NupdatedSeeds = fs->make<TH1F>("h_NupdatedSeeds","h_NupdatedSeeds",20,0,20);
    h_Eta_updatingFail = fs->make<TH1F>("h_Eta_updatingFail","h_Eta_updatingFail",60,-2.5, 2.5);
    h_Pt_updatingFail = fs->make<TH1F>("h_Pt_updatingFail","h_Pt_updatingFail",60,0,100);
    h_NrawMeas = fs->make<TH1F>("h_NrawMeas","h_NrawMeas",20,0,20);
    h_NseedLayer = fs->make<TH1F>("h_NseedLayer","h_NseedLayer",10,0,10);
  } 

}

void TSGFromPropagation::setEvent(const edm::Event& iEvent) {

  bool measTrackerChanged = false;

  unsigned long long newCacheId_MT = theService->eventSetup().get<CkfComponentsRecord>().cacheIdentifier();

  if ( newCacheId_MT != theCacheId_MT ) {
    LogTrace(theCategory) << "Measurment Tracker Geometry changed!";
    theCacheId_MT = newCacheId_MT;
    theService->eventSetup().get<CkfComponentsRecord>().get(theMeasTracker);
    measTrackerChanged = true;
  }

  theMeasTracker->update(iEvent);

  if ( measTrackerChanged && (&*theMeasTracker) ) {
     if ( theTkLayerMeasurements ) delete theTkLayerMeasurements;
     theTkLayerMeasurements = new LayerMeasurements(&*theMeasTracker);
  }

}

TrajectoryStateOnSurface TSGFromPropagation::innerState(const TrackCand& staMuon) const {

  TrajectoryStateOnSurface innerTS;

  if ( staMuon.first && staMuon.first->isValid() ) {
    if (staMuon.first->direction() == alongMomentum) {
      innerTS = staMuon.first->firstMeasurement().updatedState();
    } 
    else if (staMuon.first->direction() == oppositeToMomentum) { 
      innerTS = staMuon.first->lastMeasurement().updatedState();
    }
  } else {
    innerTS = theTSTransformer->innerStateOnSurface(*(staMuon.second),*theService->trackingGeometry(), &*theService->magneticField());
  }
  //rescale the error
  if (theErrorMatrixAdjuster && !theAdjustAtIp) adjust(innerTS);
  else innerTS.rescaleError(theErrorRescaling);

  return  innerTS;

//    return theTSTransformer->innerStateOnSurface(*(staMuon.second),*theService->trackingGeometry(), &*theService->magneticField());
}

TrajectoryStateOnSurface TSGFromPropagation::outerTkState(const TrackCand& staMuon) const {

  TrajectoryStateOnSurface result;

  if ( theUseVertexStateFlag && staMuon.second->pt() > 1.0 ) {
    FreeTrajectoryState iniState = theTSTransformer->initialFreeState(*(staMuon.second), &*theService->magneticField());
    //rescale the error at IP
    if (theErrorMatrixAdjuster && theAdjustAtIp){ adjust(iniState); }
    else iniState.rescaleError(theErrorRescaling);

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

TrajectorySeed TSGFromPropagation::createSeed(const TrajectoryStateOnSurface& tsos, const edm::OwnVector<TrackingRecHit>& container, const DetId& id) const {

  PTrajectoryStateOnDet* seedTSOS = theTSTransformer->persistentState(tsos,id.rawId());
  return TrajectorySeed(*seedTSOS,container,oppositeToMomentum);

}

/// further clean measurements
void TSGFromPropagation::selectMeasurements(std::vector<TrajectoryMeasurement>& tms) const {

  if ( tms.size() < 2 ) return;

  std::vector<bool> mask(tms.size(),true);

  std::vector<TrajectoryMeasurement> result;
  std::vector<TrajectoryMeasurement>::const_iterator iter;
  std::vector<TrajectoryMeasurement>::const_iterator jter;

  int i(0), j(0);

  for ( iter = tms.begin(); iter != tms.end(); iter++ ) {

    if ( !mask[i] ) { i++; continue; }
    j = i+1;

    for ( jter = iter+1; jter != tms.end(); jter++ ) {
      if ( !mask[j] ) { j++; continue; }
      LogTrace(theCategory) << "comparing measurment state: "<<iter->updatedState()<<"with measurment state: "<<jter->updatedState();

      if (
           ( (iter->updatedState().globalPosition() - jter->updatedState().globalPosition()).mag() < 1e-3 ) && 
           ( ( (iter->updatedState().globalMomentum() - jter->updatedState().globalMomentum()).mag() < 1.0 ) || 
               ( fabs(iter->updatedState().globalMomentum().eta() - jter->updatedState().globalMomentum().eta()) < 0.01 ) ) )  {
      
          if ( iter->estimate() > jter->estimate() ) 
            mask[i] = false;
          else mask[j] = false;
        }
      j++;
    }
   i++;
  }

  i = 0;
  for ( iter = tms.begin(); iter != tms.end(); iter++, i++ ) {
    if ( mask[i] )
      result.push_back(*iter);
  }
  tms.clear();
  tms.swap(result);

  return;

}


void TSGFromPropagation::validMeasurements(std::vector<TrajectoryMeasurement>& tms) const {

  if ( tms.empty()) return;

  std::vector<TrajectoryMeasurement> validMeas;

  // consider only valid TM
  for ( std::vector<TrajectoryMeasurement>::const_iterator measurement = tms.begin();
        measurement!= tms.end(); ++measurement ) {
    if ((*measurement).recHit()->isValid() && (*measurement).updatedState().isValid()) {
      validMeas.push_back( (*measurement) );
    }
  }
  tms.clear();
  tms.swap(validMeas);

  return;

}

std::vector<TrajectoryMeasurement> TSGFromPropagation::findMeasurements_new(const DetLayer* nl, const TrajectoryStateOnSurface& staState) const {

  std::vector<TrajectoryMeasurement> result;

  std::vector<DetLayer::DetWithState> compatDets = nl->compatibleDets(staState, *propagator(), *estimator());
  if ( compatDets.empty() )  return result;

  for (std::vector<DetLayer::DetWithState>::const_iterator idws = compatDets.begin(); idws != compatDets.end(); ++idws) {
     if ( idws->second.isValid() && (idws->first) )  {
      std::vector<TrajectoryMeasurement> tmptm = 
         theMeasTracker->idToDet(idws->first->geographicalId())->fastMeasurements(idws->second, idws->second, *propagator(), *estimator());
         result.insert(result.end(),tmptm.begin(), tmptm.end());
     }
  }
  
  validMeasurements(result);
  return result;

/*
  std::vector<TrajectoryMeasurement> updatedResult;

  for (std::vector<TrajectoryMeasurement>::const_iterator iter = result.begin(); iter != result.end(); iter++ ) {

    TrajectoryStateOnSurface updatedTSOS = updator()->update(iter->predictedState(), *(iter->recHit())); 
    if ( updatedTSOS.isValid() && (updatedTSOS.globalMomentum().perp() > 3.0) && (fabs(updatedTSOS.globalMomentum().eta() - updatedTSOS.globalPosition().eta() ) < 0.2 ) )  {
       updatedResult.push_back(TrajectoryMeasurement(iter->forwardPredictedState(), updatedTSOS, iter->recHit(), iter->estimate()));
    }
  }
  return updatedResult;
*/
}

std::vector<TrajectoryMeasurement> TSGFromPropagation::findMeasurements(const DetLayer* nl, const TrajectoryStateOnSurface& staState) const {

  std::vector<TrajectoryMeasurement> result = tkLayerMeasurements()->measurements((*nl), staState, *propagator(), *estimator());
  validMeasurements(result);
  return result;
}

void TSGFromPropagation::findSecondMeasurements(std::vector<TrajectoryMeasurement>& tms, const std::vector<const DetLayer*>& dls) const {

   std::vector<TrajectoryMeasurement> secondMeas;

   for (std::vector<TrajectoryMeasurement>::const_iterator itm = tms.begin();
        itm != tms.end(); ++itm) {
 
     TrajectoryStateOnSurface  tsos = itm->updatedState(); 
     std::vector<TrajectoryMeasurement> tmpsectm; 

     if ( !tsos.isValid() ) continue;

     for (std::vector<const DetLayer*>::const_iterator idl = dls.begin();
          idl != dls.end(); ++idl) {

            tmpsectm = findMeasurements(*idl, tsos);
            LogTrace(theCategory) << " tmpsectm again: "<<tmpsectm.size();

           if ( !tmpsectm.empty() ) {
             break;
            }
     }
      if ( !tmpsectm.empty() ) secondMeas.insert(secondMeas.end(),tmpsectm.begin(), tmpsectm.end()); 
  } 

  tms.clear();
  tms.swap(secondMeas);
  return; 
}

void TSGFromPropagation::adjust(FreeTrajectoryState & state) const {
  CurvilinearTrajectoryError oMat = state.curvilinearError();
  CurvilinearTrajectoryError sfMat = theErrorMatrixAdjuster->get(state.momentum());//FIXME with position
  MuonErrorMatrix::multiply(oMat, sfMat);
  
  state = FreeTrajectoryState(state.parameters(),
			      oMat);
}

void TSGFromPropagation::adjust(TrajectoryStateOnSurface & state) const {
  CurvilinearTrajectoryError oMat = state.curvilinearError();
  CurvilinearTrajectoryError sfMat = theErrorMatrixAdjuster->get(state.globalMomentum());//FIXME with position
  MuonErrorMatrix::multiply(oMat, sfMat);
  
  state = TrajectoryStateOnSurface(state.globalParameters(),
				   oMat,
				   state.surface(),
				   state.surfaceSide(),
				   state.weight());
}

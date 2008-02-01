#include "RecoMuon/TrackerSeedGenerator/plugins/TSGFromPropagation.h"

/** \class TSGFromPropagation
 *
 *  $Date: 2007/12/06 19:38:38 $
 *  $Revision: 1.16 $
 *  \author Chang Liu - Purdue University 
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoTracker/MeasurementDet/interface/TkStripMeasurementDet.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet & iConfig) :theTkLayerMeasurements (0), theTracker(0), theMeasTracker(0), theNavigation(0), theService(0), theEstimator(0), theConfig (iConfig)
{
}

TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet & iConfig, const MuonServiceProxy* service) : theTkLayerMeasurements (0), theTracker(0), theMeasTracker(0), theNavigation(0), theService(service),theEstimator(0), theConfig (iConfig)
{
}


TSGFromPropagation::~TSGFromPropagation()
{
  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  LogTrace(category) << " TSGFromPropagation dtor called ";

  if ( theNavigation ) delete theNavigation;
  if ( theEstimator ) delete theEstimator;
  if ( theTkLayerMeasurements ) delete theTkLayerMeasurements;
  LogTrace(category) << " TSGFromPropagation dtor finished  ";

}

void TSGFromPropagation::trackerSeeds(const TrackCand& staMuon, const TrackingRegion&, std::vector<TrajectorySeed> & result) {

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";
  MuonPatternRecoDumper debug;

  LogTrace(category) << " begin of trackerSeed ";

  TrajectoryStateOnSurface staState = innerState(staMuon);

  if ( !staState.isValid() ) { 
    LogTrace(category) << " initial state invalid, fail";
    return;
  }

  LogTrace(category) << " staState pos: "<<staState.globalPosition()
                     << " mom: "<<staState.globalMomentum() <<"eta: "<<staState.globalPosition().eta();

  staState.rescaleError(theErrorRescaling);

  std::vector<const DetLayer*> nls = theNavigation->compatibleLayers(*(staState.freeState()), oppositeToMomentum);

  LogTrace(category) << " compatible layers: "<<nls.size();

  if ( nls.empty() ) return;

  std::vector<TrajectoryMeasurement> alltm; 

  for (std::vector<const DetLayer*>::const_iterator inl = nls.begin();
       inl != nls.end(); inl++ ) {

     if ( (!alltm.empty()) || (*inl == 0) ) break;

     std::vector<TrajectoryMeasurement> tmptm = findMeasurements(*inl, staState);
     LogTrace(category) << " Number of measurements in used layer: "<<alltm.size();

     if ( tmptm.empty() )  continue;
     alltm.insert(alltm.end(),tmptm.begin(), tmptm.end());
  }

  if ( alltm.empty() ) LogTrace(category) << " NO Measurements Found: eta: "<<staState.globalPosition().eta() <<"pt "<<staState.globalMomentum().perp();

  LogTrace(category) << " Measurements for seeds: "<<alltm.size();
  selectMeasurements(alltm);
  LogTrace(category) << " Measurements for seeds after select: "<<alltm.size();

  for (std::vector<TrajectoryMeasurement>::const_iterator itm = alltm.begin();
       itm != alltm.end(); itm++) {
   LogTrace(category) << " meas: hit "<<itm->recHit()->isValid()<<" state "<<itm->updatedState().isValid() << " estimate "<<itm->estimate();

//    if ( itm->recHit()->isValid() && itm->updatedState().isValid() )  {
       TrajectorySeed ts = createSeed(*itm);
       result.push_back(ts); 
//    }
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

  theUseSecondMeasurementsFlag = theConfig.getParameter<bool>("UseSecondMeasurements");

}

void TSGFromPropagation::setEvent(const edm::Event& iEvent) {

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  // DetLayer Geometry
  bool measTrackerChanged = false;

  theService->eventSetup().get<TrackerRecoGeometryRecord>().get(theTracker);

  unsigned long long newCacheId_MT = theService->eventSetup().get<CkfComponentsRecord>().cacheIdentifier();

  if ( newCacheId_MT != theCacheId_MT ) {
    LogTrace(category) << "Measurment Tracker Geometry changed!";
    theCacheId_MT = newCacheId_MT;
    theService->eventSetup().get<CkfComponentsRecord>().get(theMeasTracker);
    measTrackerChanged = true;

  }

  theMeasTracker->update(iEvent);

  if ( measTrackerChanged && (&*theMeasTracker) ) {
     if ( theTkLayerMeasurements ) delete theTkLayerMeasurements;
     theTkLayerMeasurements = new LayerMeasurements(&*theMeasTracker);
  }

    if(theNavigation) delete theNavigation;
    theNavigation = new DirectTrackerNavigation(theTracker);

}

TrajectoryStateOnSurface TSGFromPropagation::innerState(const TrackCand& staMuon) const {

  TrajectoryStateOnSurface innerTS;
  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  if ( staMuon.first && staMuon.first->isValid() ) {

    if (staMuon.first->direction() == alongMomentum) {
      LogTrace(category)<<"alongMomentum";
      innerTS = staMuon.first->firstMeasurement().updatedState();
    } 
    else if (staMuon.first->direction() == oppositeToMomentum) { 
      LogTrace(category)<<"oppositeToMomentum";
      innerTS = staMuon.first->lastMeasurement().updatedState();
    }
    else edm::LogError(category)<<"Wrong propagation direction!";

  } else {

    TrajectoryStateTransform tsTransformer;
    innerTS = tsTransformer.innerStateOnSurface(*(staMuon.second),*theService->trackingGeometry(), &*theService->magneticField());
  }
  return  innerTS;

}

TrajectorySeed TSGFromPropagation::createSeed(const TrajectoryMeasurement& tm) const {

  TrajectoryStateOnSurface tsos = tm.updatedState(); 

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  LogTrace(category)<<"Trajectory State on Surface of Seed";
  LogTrace(category)<<tsos;

  TrajectoryStateTransform tsTransform;
    
  PTrajectoryStateOnDet* seedTSOS =
    tsTransform.persistentState(tsos,tm.recHit()->geographicalId().rawId());
    
  edm::OwnVector<TrackingRecHit> container;

  container.push_back(tm.recHit()->hit()->clone());

  return TrajectorySeed(*seedTSOS,container,oppositeToMomentum);

}

/// further clear measurements on diffrent layers
void TSGFromPropagation::selectMeasurements(std::vector<TrajectoryMeasurement>& tms) const {

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  if ( tms.size() < 2 ) return;

  vector<bool> mask(tms.size(),true);

  std::vector<TrajectoryMeasurement> result;
  std::vector<TrajectoryMeasurement>::const_iterator iter;
  std::vector<TrajectoryMeasurement>::const_iterator jter;

  int i(0), j(0);

  for ( iter = tms.begin(); iter != tms.end(); iter++ ) {

    if ( !mask[i] ) { i++; continue; }
    j = i+1;

    for ( jter = iter+1; jter != tms.end(); jter++ ) {
      if ( !mask[j] ) { j++; continue; }
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
  for ( iter = tms.begin(); iter != tms.end(); iter++ ) {

  LogTrace(category)<<"select mom eta: "<<iter->updatedState().globalMomentum().eta();
  LogTrace(category)<<"select pos eta: " << iter->updatedState().globalPosition().eta();

  LogTrace(category)<<"select delta eta: " <<fabs(iter->updatedState().globalMomentum().eta() - iter->updatedState().globalPosition().eta() );

    if ( fabs(iter->updatedState().globalMomentum().eta() - iter->updatedState().globalPosition().eta() ) > 0.2 ) continue;

    if ( mask[i] ) result.push_back(*iter);
    i++;
  }
  tms.clear();
  tms.swap(result);

  return;

}


void TSGFromPropagation::validMeasurements(std::vector<TrajectoryMeasurement>& tms) const {

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  std::vector<TrajectoryMeasurement> validMeasurements;

  // consider only valid TM
  for ( std::vector<TrajectoryMeasurement>::const_iterator measurement = tms.begin();
        measurement!= tms.end(); ++measurement ) {
    if ((*measurement).recHit()->isValid() && (*measurement).updatedState().isValid()) {
      validMeasurements.push_back( (*measurement) );
    }
  }
  tms.clear();
  tms.swap(validMeasurements);
  return;

}

std::vector<TrajectoryMeasurement> TSGFromPropagation::findMeasurements(const DetLayer* nl, const TrajectoryStateOnSurface& staState) const {

  std::vector<TrajectoryMeasurement> result;

  if ( nl == 0 ) return result;

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  MuonPatternRecoDumper debug;
  LogTrace(category) << " findMeasurements: ";
  LogTrace(category) << " this next layer: "<<debug.dumpLayer(nl);
  result =
      tkLayerMeasurements()->measurements((*nl), staState, *propagator(), *estimator());

  LogTrace(category) << "measurements: "<<result.size();
  validMeasurements(result);
  LogTrace(category) << " valid measurements: "<<result.size();

  return result;
}

void TSGFromPropagation::findSecondMeasurements(std::vector<TrajectoryMeasurement>& tms, const std::vector<const DetLayer*>& dls) const {

   std::vector<TrajectoryMeasurement> secondMeas;

   const std::string category = "Muon|RecoMuon|TSGFromPropagation";

   for (std::vector<TrajectoryMeasurement>::const_iterator itm = tms.begin();
        itm != tms.end(); ++itm) {
 
     TrajectoryStateOnSurface  tsos = itm->updatedState(); 
     std::vector<TrajectoryMeasurement> tmpsectm; 

     if ( !tsos.isValid() ) continue;

     for (std::vector<const DetLayer*>::const_iterator idl = dls.begin();
          idl != dls.end(); ++idl) {

            tmpsectm = findMeasurements(*idl, tsos);
            LogTrace(category) << " tmpsectm again: "<<tmpsectm.size();

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

#include "RecoMuon/TrackerSeedGenerator/interface/TSGFromPropagation.h"

/** \class TSGFromPropagation
 *
 *  $Date: 2007/05/21 20:31:35 $
 *  $Revision: 1.3 $
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


TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet & iConfig) :theTkLayerMeasurements (0), theTracker(0), theMeasTracker(0), theNavigation(0), theService(0), theEstimator(0), theUpdator(0), theVtxUpdator(0), theConfig (iConfig)
{
}

TSGFromPropagation::TSGFromPropagation(const edm::ParameterSet & iConfig, const MuonServiceProxy* service) : theTkLayerMeasurements (0), theTracker(0), theMeasTracker(0), theNavigation(0), theService(service),theEstimator(0), theUpdator(0), theVtxUpdator(0), theConfig (iConfig)
{
}


TSGFromPropagation::~TSGFromPropagation()
{
  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  LogTrace(category) << " TSGFromPropagation dtor called ";

  if ( theNavigation ) delete theNavigation;
  if ( theUpdator ) delete theUpdator;
  if ( theVtxUpdator ) delete theVtxUpdator;
  if ( theEstimator ) delete theEstimator;
  if ( theTkLayerMeasurements ) delete theTkLayerMeasurements;
  LogTrace(category) << " TSGFromPropagation dtor finished  ";

}

std::vector<TrajectorySeed> TSGFromPropagation::trackerSeeds(const TrackCand& staMuon, const TrackingRegion&) {

  std::vector<TrajectorySeed> result;
  const std::string category = "Muon|RecoMuon|TSGFromPropagation";
  MuonPatternRecoDumper debug;

  LogTrace(category) << " begin of trackerSeed ";

  TrajectoryStateOnSurface staState = outerTkState(staMuon);

  if ( !staState.isValid() ) staState = innerState(staMuon);

  if ( !staState.isValid() ) return result;

  LogTrace(category) << " staState pos: "<<staState.globalPosition()
                     << " mom: "<<staState.globalMomentum() <<"eta: "<<staState.globalPosition().eta();

  float err = 100;
  staState.rescaleError(err);

  std::vector<const DetLayer*> nls = theNavigation->compatibleLayers(*(staState.freeState()), oppositeToMomentum);

  LogTrace(category) << " compatible layers: "<<nls.size();

  if ( nls.empty() ) return result;

//// debug only ===========
/*
/std::vector<TkStripMeasurementDet*> stripdets = theMeasTracker->stripDets();

for (std::vector<TkStripMeasurementDet*>::const_iterator isd = stripdets.begin(); isd !=  stripdets.end(); ++isd  ) {

  TransientTrackingRecHit::RecHitContainer hits = (*isd)->recHits(staState); 
  if ( !hits.empty() ) LogTrace(category) << " here is hit ";

  for (TransientTrackingRecHit::RecHitContainer::const_iterator ihit = hits.begin(); ihit != hits.end(); ihit++ ) {
     LogTrace(category) << "a hit is at "<< (*ihit)->globalPosition();
     LogTrace(category) << "a hit is at layer "<<
     debug.dumpLayer(theTracker->detLayer((*ihit)->geographicalId()));

     TrajectoryStateOnSurface prdstat = propagator()->propagate(staState,theTracker->detLayer((*ihit)->geographicalId())->surface());

     if (prdstat.isValid() )
        LogTrace(category) << "pred tsos at layer "<<prdstat;

  }

 } 
*/
///// ======

  std::vector<TrajectoryMeasurement> alltm = findMeasurements(nls.front(), staState);
  LogTrace(category) << " allmeas first: "<<alltm.size();

  err *= 10;

  if ( alltm.empty() ) staState.rescaleError(err);

  alltm = findMeasurements(nls.front(), staState);
  LogTrace(category) << " allmeas first rescale: "<<alltm.size();

  std::vector<const DetLayer*>::iterator inl;
  std::vector<const DetLayer*>::iterator usednl;

  int iUsedLayer = 0; 

  while ( ( iUsedLayer < 3 ) && ( inl != nls.end() - 1) )  { 

     usednl = nls.begin();
     nls.erase(usednl);

     inl = nls.begin();

     if (nls.size() < 10 ) break;
     if ( inl == nls.end() - 1 ) break;
     if ( *inl == 0 ) break;

     std::vector<TrajectoryMeasurement> tmptm = findMeasurements(*inl, staState);
     LogTrace(category) << " Number of measurements in used layer: "<<iUsedLayer<<" is" <<alltm.size();
     iUsedLayer++;

     if ( !tmptm.empty() ) { 
       iUsedLayer++;
       alltm.insert(alltm.end(),tmptm.begin(), tmptm.end());
     } else  {
        err *= 10; 
        staState.rescaleError(err);
     }

  }

   LogTrace(category) << " remaining layers: "<<nls.size();

  if ( alltm.empty() ) LogTrace(category) << " NO Measurements Found: eta: "<<staState.globalPosition().eta() <<"pt "<<staState.globalMomentum().perp();

  findSecondMeasurements(alltm, nls);

  selectMeasurements(alltm);
   LogTrace(category) << " Measurements for seeds: "<<alltm.size();

  for (std::vector<TrajectoryMeasurement>::const_iterator itm = alltm.begin();
       itm != alltm.end(); itm++) {
   LogTrace(category) << " meas: hit "<<itm->recHit()->isValid()<<" state "<<itm->updatedState().isValid() << " estimate "<<itm->estimate();

    if ( itm->recHit()->isValid() && itm->updatedState().isValid() )  {
      LogTrace(category) << " create seed ";

       TrajectorySeed ts = createSeed(*itm);
       result.push_back(ts); 
    }
  }

  return result;

}
    

void TSGFromPropagation::init(const MuonServiceProxy* service) {

  theMaxChi2 = theConfig.getParameter<double>("MaxChi2");
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2);

  theUpdator= new KFUpdator();
  theCacheId_MT = 0;

  thePropagatorName = theConfig.getParameter<std::string>("Propagator");

  theService = service;

  edm::ParameterSet vtxUpdatorParameters = theConfig.getParameter<edm::ParameterSet>("MuonUpdatorAtVertexParameters");

  theVtxUpdator = new MuonUpdatorAtVertex(vtxUpdatorParameters,theService);


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

TrajectoryStateOnSurface TSGFromPropagation::outerTkState(const TrackCand& staMuon) const {

  const string category = "Muon|RecoMuon|TSGFromPropagation";
  MuonPatternRecoDumper debug;
 
  // build the transient track
  reco::TransientTrack transientTrack(staMuon.second,
				      &*theService->magneticField(),
				      theService->trackingGeometry());

  LogTrace(category) << "Apply the vertex constraint";
  pair<bool,FreeTrajectoryState> updateResult = theVtxUpdator->update(transientTrack);

  if(!updateResult.first){
    LogTrace(category) << "vertex constraint failed ";
    return TrajectoryStateOnSurface(); //FIXME
  }

  LogTrace(category) << "FTS after the vertex constraint";
  FreeTrajectoryState &ftsAtVtx = updateResult.second;

  LogTrace(category) << debug.dumpFTS(ftsAtVtx);

  StateOnTrackerBound fromInside(&*theService->propagator("PropagatorWithMaterial"));

  TrajectoryStateOnSurface result = fromInside(ftsAtVtx);

  return result;

}


TrajectorySeed TSGFromPropagation::createSeed(const TrajectoryMeasurement& tm) const {

  TrajectoryStateOnSurface tsos = tm.updatedState(); 

  const std::string category = "Muon|RecoMuon|TSGFromPropagation";

  LogTrace(category)<<"Trajectory State on Surface of Seed";
  LogTrace(category)<<"original seed TSOS: " << tsos;

  resetError(tsos);

  LogTrace(category)<<"reseted seed TSOS: " << tsos;


/*
  LogTrace(category) << "createSeed: Apply the vertex constraint";
  pair<bool,FreeTrajectoryState> updateResult = theVtxUpdator->update(*(tsos.freeState()));

  if(!updateResult.first){
    LogTrace(category) << "createSeed: vertex constraint failed ";
  } else {

    LogTrace(category) << "FTS after the vertex constraint";
    FreeTrajectoryState &ftsAtVtx = updateResult.second;
    LogTrace(category) << ftsAtVtx;
    tsos = theService->propagator("PropagatorWithMaterial")->propagate(ftsAtVtx, tm.layer()->surface());

  }
*/

  TrajectoryStateTransform tsTransform;
    
  PTrajectoryStateOnDet *seedTSOS =
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

  if (tms.size() > 5 ) {
    std::stable_sort(tms.begin(),tms.end(),IncreasingEstimate());
    tms.erase(tms.begin()+5, tms.end());
   }

  return;

}


void TSGFromPropagation::validMeasurements(std::vector<TrajectoryMeasurement>& tms) const {

  const std::string category = "Muon|RecoMuon|MuonTkTrajectoryBuilder";

  std::vector<TrajectoryMeasurement> validMeasurements;

  // consider only valid TM
  for ( std::vector<TrajectoryMeasurement>::const_iterator measurement = tms.begin();
        measurement!= tms.end(); ++measurement ) {
    if ((*measurement).recHit()->isValid() && (*measurement).predictedState().isValid()) {
      validMeasurements.push_back( (*measurement) );
    }
  }
  tms.clear();
  tms.swap(validMeasurements);
  return;

/*
  std::vector<TrajectoryMeasurement> bestMeasurements;
  float dchi2 = 9999.0;
  std::vector<TrajectoryMeasurement>::const_iterator theBestMeasurement = validMeasurements.end();
  for (std::vector<TrajectoryMeasurement>::const_iterator measurement = validMeasurements.begin(); measurement!= validMeasurements.end(); measurement++ ) {
    TrajectoryStateOnSurface tsos = (measurement)->updatedState();
    if ( !tsos.isValid() ) tsos = (measurement)->predictedState();
    if ( !tsos.isValid() ) continue;
    LogTrace(category)<<"estimate: "<<(measurement)->estimate();
    if (fabs(tsos.globalPosition().eta() - tsos.globalMomentum().eta() ) > 0.2 )   {
    LogTrace(category)<<"best measurements: traj direction too off, skip ...";
    continue;
   }
    TransientTrackingRecHit::ConstRecHitPointer recHit = (measurement)->recHit();
    if ( !recHit->isValid() ) continue;
    std::pair<bool,double> chi2 = theEstimator->estimate(tsos, *(recHit));
    if ( chi2.first && (chi2.second < dchi2) ) {
      dchi2 = chi2.second;
      theBestMeasurement = measurement;
    }
  }

  if ( theBestMeasurement != validMeasurements.end() ) bestMeasurements.push_back(*theBestMeasurement);
  tms.clear();
  tms.swap(bestMeasurements);
*/

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
     if ( tmpsectm.empty() ) secondMeas.push_back(*itm);
     else secondMeas.insert(secondMeas.end(),tmpsectm.begin(), tmpsectm.end()); 
  } 

  tms.clear();
  tms.swap(secondMeas);
  return; 
}

void TSGFromPropagation::resetError(TrajectoryStateOnSurface& tsos) const {

   AlgebraicSymMatrix55 matrix = AlgebraicMatrixID();

   matrix(0,0) = 0.01; //charge/momentum
   matrix(1,1) = 0.02; //lambda
   matrix(2,2) = 0.05; // phi
   matrix(3,3) = 10.0; //x
   matrix(4,4) = 10.0; //y

   CurvilinearTrajectoryError error(matrix);
 
   TrajectoryStateOnSurface newTsos(tsos.globalParameters(), error, tsos.surface()); 
   tsos = newTsos;
   return;

}

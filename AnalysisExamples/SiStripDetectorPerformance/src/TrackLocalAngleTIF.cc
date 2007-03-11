
#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngleTIF.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrackFitters/interface/KFFittingSmoother.h"

using namespace std;
TrackLocalAngleTIF::TrackLocalAngleTIF(edm::ParameterSet const& conf) : 
  conf_(conf)
{
}
void TrackLocalAngleTIF::init(const edm::Event& e, const edm::EventSetup& es){

  //
  // get geometry
  //
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  tracker=&(* estracker);
  //
  // get magnetic field
  //
  edm::ESHandle<MagneticField> esmagfield;
  es.get<IdealMagneticFieldRecord>().get(esmagfield);
  magfield=&(*esmagfield);


// Check this, since it takes theFitter if it is not an MTCCtrack
// but later it recreates it.
// Remove candidate
// ----------------
  //
  // get the fitter
  //
//   if(!(conf_.getParameter<bool>("MTCCtrack"))){
//     edm::ESHandle<TrajectoryFitter> fitter;
//     LogDebug("TrackLocalAngleTIF") << "get the fitter from the ES" << "\n";
//     std::string fitterName = conf_.getParameter<std::string>("Fitter");   
//     es.get<TrackingComponentsRecord>().get(fitterName,fitter);
//     theFitter=&(*fitter);

//     //
//     // get also the propagator
//     //
//     edm::ESHandle<Propagator> propagator;
//     LogDebug("TrackLocalAngleTIF") << "get also the propagator" << "\n";
//     std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
//     es.get<TrackingComponentsRecord>().get(propagatorName,propagator);
//     thePropagator=&(*propagator);
//   }
// ----------------


  //
  // get the builder
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> builder;
  LogDebug("TrackLocalAngleTIF") << "get also the TransientTrackingRecHitBuilder" << "\n";
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,builder);
  RHBuilder=&(*builder);

  // Clean up Angles vectors
  oXZHitAngle.clear();
  oYZHitAngle.clear();

  oLocalDir.clear();
  oGlobalDir.clear();
}

// Virtual destructor needed.
TrackLocalAngleTIF::~TrackLocalAngleTIF() {  
}  

std::vector<std::pair<const TrackingRecHit*,float> > TrackLocalAngleTIF::findtrackangle(const reco::Track& theT)
{
  //  int cont = 0;
  //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
  //meanwhile computes the number of degrees of freedom
  LogDebug("TrackLocalAngleTIF") << "Start\n";
  TransientTrackingRecHit::RecHitContainer tmp;
  TransientTrackingRecHit::RecHitContainer hits;
  
  for (trackingRecHit_iterator i=theT.recHitsBegin();
       i!=theT.recHitsEnd(); i++){
    // 	hits.push_back(builder->build(&**i ));
    // 	  if ((*i)->isValid()){
    tmp.push_back(RHBuilder->build(&**i ));
  }
  LogDebug("TrackLocalAngleTIF") << "Transient rechit filled" << "\n";
  
  //SORT RECHITS ALONGMOMENTUM
  const TransientTrackingRecHit::ConstRecHitPointer *firstHit;
  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.begin(); it!=tmp.end();it++){
    if ((**it).isValid()) {
      firstHit = &(*it);
      break;
    }
  }
  const TransientTrackingRecHit::ConstRecHitPointer *lastHit;
  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1; it!=tmp.begin()-1;it--){
    if ((**it).isValid()) {
      lastHit= &(*it);
      break;
    }
  }
  if ((*firstHit)->globalPosition().mag2() > ((*lastHit)->globalPosition().mag2()) ){
    //FIXME temporary should use reverse
    for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1;it!=tmp.begin()-1;it--){
      hits.push_back(*it);
    }
  } else hits=tmp;
  
  reco::TransientTrack theTT(theT, thePropagator->magneticField() );
  
  //       TrajectoryStateOnSurface theTSOS=theTT.impactPointState();
  //       theTSOS.rescaleError(100);
  
  TrajectoryStateOnSurface firstState=thePropagator->propagate(theTT.impactPointState(), hits.front()->det()->surface());
  AlgebraicSymMatrix C(5,1);
  C *= 100.;
  TrajectoryStateOnSurface theTSOS( firstState.localParameters(), LocalTrajectoryError(C),
				    firstState.surface(),
				    thePropagator->magneticField()); 
  
  LogDebug("TrackLocalAngleTIF") << "Initial TSOS\n" << theTSOS << "\n";
  
  const TrajectorySeed * seed = new TrajectorySeed();//empty seed: not needed
  //buildTrack
  return buildTrack(hits, theTSOS, *seed);
  //  return buildTrack(theTSOS, *seed);
}

std::vector< std::pair<const TrackingRecHit*,float> > TrackLocalAngleTIF::buildTrack(
										     TransientTrackingRecHit::RecHitContainer& hits,
										     const TrajectoryStateOnSurface& theTSOS,
										     const TrajectorySeed& seed)
{
  //variable declarations
  std::vector<Trajectory> trajVec;
  std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
  Trajectory * theTraj; 
  
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
  trajVec = theFitter->fit(seed, hits, theTSOS);
  
  LogDebug("TrackProducer") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  
  if (trajVec.size() != 0){

    theTraj = new Trajectory( trajVec.front() );
    
    LogDebug("TrackLocalAngleTIF") <<"track done";
    std::vector<TrajectoryMeasurement> TMeas=theTraj->measurements();

    std::vector<TrajectoryMeasurement>::iterator itm;
    int i=0;
    LogDebug("TrackLocalAngleTIF::findtrackangle")<<"Loop on rechit and TSOS";
    for (itm=TMeas.begin();itm!=TMeas.end();itm++){
      //std::cout<<"hit: "<<i++<<std::endl;
      TrajectoryStateOnSurface tsos=itm->updatedState();
      const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
      LocalVector trackdirection=tsos.localDirection();
      if(matchedhit){//if matched hit...
	
	GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
	
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	std::cout<<"Track direction trasformed in global direction"<<std::endl;
	
	//cluster and trackdirection on mono det
	
	// THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *monohit=matchedhit->monoHit();
	    
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > monocluster=monohit->cluster();
	const GeomDetUnit * monodet=gdet->monoDet();
	
	LocalVector monotkdir=monodet->toLocal(gtrkdir);
	//size=(monocluster->amplitudes()).size();
	if(monotkdir.z()!=0){
	  
	  // THE LOCAL ANGLE (MONO)
	  float angle = atan(monotkdir.x()/monotkdir.z())*180/TMath::Pi();
	  //
	  hitangleassociation.push_back(make_pair(monohit, angle)); 
	  oXZHitAngle.push_back( make_pair( monohit, atan( monotkdir.x() / monotkdir.z())));
	  oYZHitAngle.push_back( make_pair( monohit, atan( monotkdir.y() / monotkdir.z())));
	  oLocalDir.push_back( make_pair( monohit, monotkdir));
	  oGlobalDir.push_back( make_pair( monohit, gtrkdir));
	  //	  std::cout<<"Angle="<<atan(monotkdir.x()/monotkdir.z())*180/TMath::Pi()<<std::endl;

	    
	    
	  //cluster and trackdirection on stereo det
	    
	  // THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	  const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	   
		
	  const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
	  const GeomDetUnit * stereodet=gdet->stereoDet(); 
	  LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	  //size=(stereocluster->amplitudes()).size();
	  if(stereotkdir.z()!=0){
	    
	    // THE LOCAL ANGLE (STEREO)
	    float angle = atan(stereotkdir.x()/stereotkdir.z())*180/TMath::Pi();
	    hitangleassociation.push_back(make_pair(stereohit, angle)); 
	    oXZHitAngle.push_back( make_pair( stereohit, atan( stereotkdir.x() / stereotkdir.z())));
	    oYZHitAngle.push_back( make_pair( stereohit, atan( stereotkdir.y() / stereotkdir.z())));
	    oLocalDir.push_back( make_pair( stereohit, stereotkdir));
	    oGlobalDir.push_back( make_pair( stereohit, gtrkdir));
	  }
	}
      }
      else if(hit){
	//  hit= POINTER TO THE RECHIT
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	//size=(cluster->amplitudes()).size();
	  
	
	if(trackdirection.z()!=0){
	  
	  // THE LOCAL ANGLE (STEREO)
	  float angle = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
	  hitangleassociation.push_back(make_pair(hit, angle)); 
	  oXZHitAngle.push_back( make_pair( hit, atan( trackdirection.x() / trackdirection.z())));
	  oYZHitAngle.push_back( make_pair( hit, atan( trackdirection.y() / trackdirection.z())));
	  oLocalDir.push_back( make_pair( hit, trackdirection));
	  oGlobalDir.push_back( make_pair( hit, tsos.globalDirection()));
	}
      }
      LogDebug("TrackLocalAngleTIF")<<"I found "<<i<<" hits.";
    }
  }
  return hitangleassociation;
}


std::vector<std::pair<const TrackingRecHit*,float> > TrackLocalAngleTIF::findtrackangle(const TrajectorySeed& seed,
											const reco::Track& theT){
  std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
  seed_plus=(seed.direction()==alongMomentum);
  //services  
  LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"Start find track angle";
  if (seed_plus) { 	 
    thePropagator= new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
    thePropagatorOp= new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) );} 	 
  else {
    thePropagator=      new PropagatorWithMaterial(oppositeToMomentum,0.1057,&(*magfield) ); 	
    thePropagatorOp=    new PropagatorWithMaterial(alongMomentum,0.1057,&(*magfield) );
  }
  
  theUpdator=       new KFUpdator();
  theEstimator=     new Chi2MeasurementEstimator(30);
  
  LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"Now construct the KF fitters";
  
  const KFTrajectoryFitter theKFFitter=        KFTrajectoryFitter(*thePropagator,
								  *theUpdator,	
								  *theEstimator) ;
  const KFTrajectorySmoother theKFSmoother=      KFTrajectorySmoother(*thePropagatorOp,
								      *theUpdator,	
								      *theEstimator);
  LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"Contructing Trajectory State of seeds";
  theFitter = new KFFittingSmoother(theKFFitter,theKFSmoother);
  const TrajectoryStateOnSurface  startingState=startingTSOS(seed);
  
  
  TransientTrackingRecHit::RecHitContainer tmp;
  TransientTrackingRecHit::RecHitContainer hits;
  
  for (trackingRecHit_iterator i=theT.recHitsBegin();
       i!=theT.recHitsEnd(); i++){
    // 	hits.push_back(builder->build(&**i ));
    // 	  if ((*i)->isValid()){
    tmp.push_back(RHBuilder->build(&**i ));
    //	  }
  }
  
  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1;it!=tmp.begin()-1;it--){
    hits.push_back(*it);
  }

  return buildTrack(hits, startingState, seed);
}       


TrajectoryStateOnSurface   TrackLocalAngleTIF::startingTSOS(const TrajectorySeed& seed)const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = (&(*tracker))->idToDet(DetId(pState.detId()));
  TrajectoryStateOnSurface State= tsTransform.transientState( pState, &(gdet->surface()), 
							      &(*magfield));
  return State;

}



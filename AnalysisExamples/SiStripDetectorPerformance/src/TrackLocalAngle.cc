#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngle.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalVector.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
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
TrackLocalAngle::TrackLocalAngle(edm::ParameterSet const& conf) : 
  conf_(conf)
{
}
void TrackLocalAngle::init(const edm::Event& e, const edm::EventSetup& es){

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

  //
  // get the fitter
  //
  if(!(conf_.getParameter<bool>("MTCCtrack"))){
    edm::ESHandle<TrajectoryFitter> fitter;
    LogDebug("TrackLocalAngle") << "get the fitter from the ES" << "\n";
    std::string fitterName = conf_.getParameter<std::string>("Fitter");   
    es.get<TrackingComponentsRecord>().get(fitterName,fitter);
    theFitter=&(*fitter);

    //
  // get also the propagator
  //
    edm::ESHandle<Propagator> propagator;
    LogDebug("TrackLocalAngle") << "get also the propagator" << "\n";
    std::string propagatorName = conf_.getParameter<std::string>("Propagator");   
    es.get<TrackingComponentsRecord>().get(propagatorName,propagator);
    thePropagator=&(*propagator);
  }
  //
  // get the builder
  //
  edm::ESHandle<TransientTrackingRecHitBuilder> builder;
  LogDebug("TrackLocalAngle") << "get also the TransientTrackingRecHitBuilder" << "\n";
  std::string builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  es.get<TransientRecHitRecord>().get(builderName,builder);
  RHBuilder=&(*builder);
}

// Virtual destructor needed.
TrackLocalAngle::~TrackLocalAngle() {  
}  

std::vector<std::pair<const TrackingRecHit*,float> > TrackLocalAngle::findtrackangle(const reco::Track& theT)
{
  //  int cont = 0;
    //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
    //meanwhile computes the number of degrees of freedom
  LogDebug("TrackLocalAngle") << "Start\n";


  TransientTrackingRecHit::RecHitContainer tmp;
  TransientTrackingRecHit::RecHitContainer hits;

  
  for (trackingRecHit_iterator i=theT.recHitsBegin();
       i!=theT.recHitsEnd(); i++){
    // 	hits.push_back(builder->build(&**i ));
    // 	  if ((*i)->isValid()){
    tmp.push_back(RHBuilder->build(&**i ));
  }

  cout << "almeno qui ci arrivi ? " << endl;

   LogDebug("TrackLocalAngle") << "Transient rechit filled" << "\n";
  
  //SORT RECHITS ALONGMOMENTUM
  const TransientTrackingRecHit::ConstRecHitPointer *firstHit=0;
  for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.begin(); it!=tmp.end();it++){
    if ((**it).isValid()) {
      firstHit = &(*it);
      break;
    }
  }
  const TransientTrackingRecHit::ConstRecHitPointer *lastHit=0;
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
  
  LogDebug("TrackLocalAngle") << "Initial TSOS\n" << theTSOS << "\n";
  
  const TrajectorySeed * seed = new TrajectorySeed();//empty seed: not needed
  //buildTrack
  return buildTrack(hits, theTSOS, *seed);
  //  return buildTrack(theTSOS, *seed);
}

std::vector< std::pair<const TrackingRecHit*,float> > TrackLocalAngle::buildTrack(
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
    
    LogDebug("TrackLocalAngle") <<"track done";
    std::vector<TrajectoryMeasurement> TMeas=theTraj->measurements();

    std::vector<TrajectoryMeasurement>::iterator itm;
    int i=0;
    LogDebug("TrackLocalAngle::findtrackangle")<<"Loop on rechit and TSOS";
    for (itm=TMeas.begin();itm!=TMeas.end();itm++){
      //std::cout<<"hit: "<<i++<<std::endl;
      TrajectoryStateOnSurface tsos=itm->updatedState();
      const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
      const SiPixelRecHit* pixelhit=dynamic_cast<const SiPixelRecHit*>((*thit).hit()); // [added by Andrea]
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
	  //	  float angle = atan(monotkdir.x()/monotkdir.z())*180/TMath::Pi();
 	  float angle=acos(cosineangle(monotkdir)); // [modified by Andrea]
	  //
	  hitangleassociation.push_back(make_pair(monohit, angle)); 
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
	    //		  float angle = atan(stereotkdir.x()/stereotkdir.z())*180/TMath::Pi();
 	    float angle=acos(cosineangle(stereotkdir)); // [modified by Andrea]
	    hitangleassociation.push_back(make_pair(stereohit, angle)); 
		  
	  }
	}
      }
      else if(hit){
	//  hit= POINTER TO THE RECHIT
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	//size=(cluster->amplitudes()).size();
	  
	
	if(trackdirection.z()!=0){
	  
	    // THE LOCAL ANGLE (STEREO)
	  //	  float angle = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
 	  float angle=acos(cosineangle(trackdirection)); // [modified by Andrea]
	  hitangleassociation.push_back(make_pair(hit, angle)); 
	}
      }
      else if(pixelhit){ // [added by Andrea]
 	const edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster, edm::refhelper::FindForDetSetVector<SiPixelCluster> > cluster=pixelhit->cluster();	
 	float angle=acos(cosineangle(trackdirection));
 	//	cout << "angle = " << angle << endl;
 	hitangleassociation.push_back(make_pair(pixelhit, angle)); 
      }

      LogDebug("TrackLocalAngle")<<"I found "<<i<<" hits.";
    }
  }
  return hitangleassociation;
}


std::vector<std::pair<const TrackingRecHit*,float> > TrackLocalAngle::findtrackangle(const TrajectorySeed& seed,

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
  //  TrajectoryStateOnSurface  startingState=startingTSOS(seed);
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


TrajectoryStateOnSurface   TrackLocalAngle::startingTSOS(const TrajectorySeed& seed)const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = (&(*tracker))->idToDet(DetId(pState.detId()));
  TrajectoryStateOnSurface State= tsTransform.transientState( pState, &(gdet->surface()), 
							      &(*magfield));
  return State;

}


double TrackLocalAngle::cosineangle(LocalVector trackdirection) // [added by Andrea]
{
  // z: along module thickness, x: along shortest module dimension, y: along longest module direction
  double cosineangle=0.;
  cosineangle = trackdirection.z()/trackdirection.mag();
  return cosineangle;
}


#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "CalibTracker/SiPixelLorentzAngle/interface/TrackLocalAngle.h"

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
TrackLocalAngle::~TrackLocalAngle() {  }  

std::vector<std::pair<const TrackingRecHit*,float> > TrackLocalAngle::findtrackangle(const reco::Track& theT)
{
  //  int cont = 0;
    //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
    //meanwhile computes the number of degrees of freedom
	LogDebug("TrackLocalAngle") << "Start\n";
	
	std::vector<Trajectory> trajVec;
	std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
 	Trajectory * theTraj; 
	
	trajVec = buildTrajectory(theT);
	
	LogDebug("TrackProducer") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  
	TrajectoryStateOnSurface innertsos;
  
	if (trajVec.size() != 0){

		theTraj = new Trajectory( trajVec.front() );
    
		if (theTraj->direction() == alongMomentum) {
			innertsos = theTraj->firstMeasurement().updatedState();
		} else { 
			innertsos = theTraj->lastMeasurement().updatedState();
		}
    
		LogDebug("TrackLocalAngle") <<"track done";
		std::vector<TrajectoryMeasurement> TMeas=theTraj->measurements();

		vector<TrajectoryMeasurement>::iterator itm;
		int i=0;
		LogDebug("TrackLocalAngle::findtrackangle")<<"Loop on rechit and TSOS";
		for (itm=TMeas.begin();itm!=TMeas.end();itm++){
// 			std::cout<<"hit: "<<i++<<std::endl;
			TrajectoryStateOnSurface tsos=itm->updatedState();
			const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
			const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
			const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
			LocalVector trackdirection=tsos.localDirection();
			if(matchedhit){//if matched hit...
	
				GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
				
				GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
// 				std::cout<<"Track direction trasformed in global direction"<<std::endl;
				
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
				}
			}
			LogDebug("TrackLocalAngle")<<"I found "<<i<<" hits.";
		}
	}
	return hitangleassociation;
}

std::vector<std::pair<const TrackingRecHit*,TrackLocalAngle::Trackhit> > TrackLocalAngle::findPixelParameters(const reco::Track& theT)
{
	LogDebug("TrackLocalAngle") << "Start\n";
	
	std::vector<Trajectory> trajVec;
	std::vector<std::pair<const TrackingRecHit*,Trackhit> >hitangleassociation;
	Trajectory * theTraj; 
	
	trajVec = buildTrajectory(theT);
	
	LogDebug("TrackProducer") <<" FITTER FOUND "<< trajVec.size() << " TRAJECTORIES" <<"\n";
  
	TrajectoryStateOnSurface innertsos;
  
	if (trajVec.size() != 0){

		theTraj = new Trajectory( trajVec.front() );
    
		if (theTraj->direction() == alongMomentum) {
			innertsos = theTraj->firstMeasurement().updatedState();
		} else { 
			innertsos = theTraj->lastMeasurement().updatedState();
		}
    
		LogDebug("TrackLocalAngle") <<"track done";
		std::vector<TrajectoryMeasurement> TMeas=theTraj->measurements();

		vector<TrajectoryMeasurement>::iterator itm;
		int i=0;
		LogDebug("TrackLocalAngle::findtrackangle")<<"Loop on rechit and TSOS";
		for (itm=TMeas.begin();itm!=TMeas.end();itm++){
// 			std::cout<<"hit: "<<i++<<std::endl;
			TrajectoryStateOnSurface tsos=itm->updatedState();
			const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
			const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>((*thit).hit());
			LocalVector trackdirection=tsos.localDirection();
			LocalPoint trackposition=tsos.localPosition();
			if(rechit){
				//  hit= POINTER TO THE RECHIT
				
				
				if(trackdirection.z()!=0){
					
						// THE LOCAL ANGLE (STEREO)
					Trackhit trackhit_;
					trackhit_.alpha = atan2(trackdirection.z(),trackdirection.x());
					trackhit_.beta = atan2(trackdirection.z(),trackdirection.y());
					trackhit_.gamma = atan2(trackdirection.x(),trackdirection.y());
					trackhit_.x = trackposition.x();
					trackhit_.y = trackposition.y();
					hitangleassociation.push_back(make_pair(rechit, trackhit_)); 
				}
			}
			LogDebug("TrackLocalAngle")<<"I found "<<i<<" hits.";
		}
// 		delete theTraj;
// 		theTraj = 0;
	}
	return hitangleassociation;
}

std::vector<Trajectory> TrackLocalAngle::buildTrajectory(const reco::Track& theT)
{
	TransientTrackingRecHit::RecHitContainer tmp;
	TransientTrackingRecHit::RecHitContainer hits;
  
	float ndof=0;
  
	for (trackingRecHit_iterator i=theT.recHitsBegin(); i!=theT.recHitsEnd(); i++){
    // 	hits.push_back(builder->build(&**i ));
    // 	  if ((*i)->isValid()){
		tmp.push_back(RHBuilder->build(&**i ));
		if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
    //	  }
	}
	LogDebug("TrackLocalAngle") << "Transient rechit filled" << "\n";
  
	ndof = ndof - 5;
  
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
  
	LogDebug("TrackLocalAngle") << "Initial TSOS\n" << theTSOS << "\n";
  
	const TrajectorySeed * seed = new TrajectorySeed();//empty seed: not needed
  		  
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
	return theFitter->fit(*seed, hits, theTSOS);
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
  
  theFitter=        new KFTrajectoryFitter(*thePropagator,
					   *theUpdator,	
					   *theEstimator) ;
  theSmoother=      new KFTrajectorySmoother(*thePropagatorOp,
					     *theUpdator,	
					     *theEstimator);
  LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"Contructing Trajectory State of seeds";
  
  const TrajectoryStateOnSurface  startingState=startingTSOS(seed);
  
  //  if (seed_plus) stable_sort(hits.begin(),hits.end(),CompareHitY_plus(*tracker));
  //  edm::OwnVector<TransientTrackingRecHit> tmp_trans_hits;
  //edm::OwnVector<const TransientTrackingRecHit> trans_hits;
    
  TransientTrackingRecHit::RecHitContainer tmp;
  TransientTrackingRecHit::RecHitContainer hits;
  
//   float ndof=0;
  for (trackingRecHit_iterator i=theT.recHitsBegin();
       i!=theT.recHitsEnd(); i++){
    // 	hits.push_back(builder->build(&**i ));
    // 	  if ((*i)->isValid()){
    tmp.push_back(RHBuilder->build(&**i ));
    //	  }
  }
  
  
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
  //if ((*firstHit)->globalPosition().mag2() > ((*lastHit)->globalPosition().mag2()) ){
    //FIXME temporary should use reverse
    for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.end()-1;it!=tmp.begin()-1;it--){
      hits.push_back(*it);
    }
    //} else hits=tmp;
  

 //  TransientTrackingRecHit::RecHitContainer trans_hits;
  
//   for (unsigned int icosmhit=hits->size()-1;icosmhit>0;icosmhit--){
//    const TransientTrackingRecHit::RecHitPointer tmphit=RHBuilder->build(&((*hits)[icosmhit]));
//     //tmp_trans_hits.push_back(&(*tmphit));
//     trans_hits.push_back(&(*tmphit));
    
//   }
//   const TransientTrackingRecHit::RecHitPointer tmphit=RHBuilder->build(&((*hits)[0]));
//   //tmp_trans_hits.push_back(&(*tmphit));
//   trans_hits.push_back(&(*tmphit));
//    //const  edm::OwnVector<const TransientTrackingRecHit> trans_hits(tmp_trans_hits);
//   //  for (edm::OwnVector<const TransientTrackingRecHit>::const_iterator itp=trans_hits.begin();
//   //    itp!=trans_hits.end();itp++)  cout<<(*itp).globalPosition()<<endl;
  
  LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"Start fitting";
  
  std::vector<Trajectory> Traj1;
  std::vector<Trajectory> Traj2;
  Traj1 = theFitter->fit(seed, hits,startingState);
  
  if (Traj1.size() != 0){
    const Trajectory ifitted=(Traj1.front());
    // cout<<"CHI2 "<<ifitted.chiSquared()<<endl;
    Traj2 = theSmoother->trajectories(ifitted);  
    if (Traj2.size() !=0){
      const Trajectory ismoothed=(Traj2.front());
      // cout<<"CHI3 "<<ismoothed.chiSquared()<<endl;
      
      LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"End fitting";
  
      std::vector<TrajectoryMeasurement> TMeas=ismoothed.measurements();
      //  cout<<"TM "<<TMeas.size()<<endl;

      vector<TrajectoryMeasurement>::iterator itm;
//       int i=0;
      LogDebug("AnalyzeMTCCTracks::findtrackangle")<<"Loop on rechit and TSOS";
      for (itm=TMeas.begin();itm!=TMeas.end();itm++){
// 				std::cout<<"hit: "<<i++<<std::endl;
				TrajectoryStateOnSurface tsos=itm->updatedState();
				const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
				//	TrackingRecHitCollection::const_iterator rhiterator;
				//TrackingRecHitCollection::const_iterator righthit;
			// 	for(rhiterator=hits->begin();rhiterator!=hits->end();rhiterator++){
			// 	  if (((*thit).hit()->geographicalId()).rawId()==(rhiterator->geographicalId()).rawId()){
			// 	    rhiterator=righthit;
			// 	  }
					
			//	}
				//const SiStripMatchedRecHit2D* matchedhit=(&(*righthit));
       	const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>((*thit).hit());
				const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>((*thit).hit());
				LocalVector trackdirection=tsos.localDirection();
				if(matchedhit){//if matched hit...
			
					GluedGeomDet * gdet=(GluedGeomDet *)tracker->idToDet(matchedhit->geographicalId());
					
					GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
// 					std::cout<<"Track direction trasformed in global direction"<<std::endl;
					
					//cluster and trackdirection on mono det
					
					const SiStripRecHit2D *monohit=matchedhit->monoHit();
			
					const GeomDetUnit * monodet=gdet->monoDet();
			
					LocalVector monotkdir=monodet->toLocal(gtrkdir);
			
					if(monotkdir.z()!=0){
						float angle = atan(monotkdir.x()/monotkdir.z())*180/TMath::Pi();
						hitangleassociation.push_back(make_pair(monohit, angle)); 
			
					}
					
					//cluster and trackdirection on stereo det
					
					const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
			
					const GeomDetUnit * stereodet=gdet->stereoDet(); 
			
					LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
			
					if(stereotkdir.z()!=0){
						float angle = atan(stereotkdir.x()/stereotkdir.z())*180/TMath::Pi();
						hitangleassociation.push_back(make_pair(stereohit, angle)); 
					}
					
				}
	
				else if(hit){
					
					if(trackdirection.z()!=0){
						float angle = atan(trackdirection.x()/trackdirection.z())*180/TMath::Pi();
						hitangleassociation.push_back(make_pair(hit, angle)); 	    
					}
				}  
	
      }
//       std::cout<<"I found "<<i<<" hits."<<std::endl;
      
    }
  }
  //cout<<"Chi Square = "<<chi2<<endl;
  return hitangleassociation;
}       


TrajectoryStateOnSurface   TrackLocalAngle::startingTSOS(const TrajectorySeed& seed)const
{
  PTrajectoryStateOnDet pState( seed.startingState());
  const GeomDet* gdet  = (&(*tracker))->idToDet(DetId(pState.detId()));
  TrajectoryStateOnSurface State= tsTransform.transientState( pState, &(gdet->surface()), 
							      &(*magfield));
  return State;
}



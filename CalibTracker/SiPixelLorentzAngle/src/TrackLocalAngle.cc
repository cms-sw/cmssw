
#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>
#include "CalibTracker/SiPixelLorentzAngle/interface/TrackLocalAngle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
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
 
typedef edm::OwnVector<TrackingRecHit> recHitContainer;
 
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

std::vector<std::pair<SiPixelRecHit*,TrackLocalAngle::Trackhit> > TrackLocalAngle::findPixelParameters(const reco::Track& theT)
{
	LogDebug("TrackLocalAngle") << "Start\n";
	
	std::vector<Trajectory> trajVec;
	std::vector<std::pair<SiPixelRecHit*,Trackhit> >hitangleassociation;
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
			TrajectoryStateOnSurface tsos=itm->updatedState();
			const TransientTrackingRecHit::ConstRecHitPointer thit=itm->recHit();
			const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>((*thit).hit());
			LocalVector trackdirection=tsos.localDirection();
			LocalPoint trackposition=tsos.localPosition();
			if(rechit){
				//  rechit= POINTER TO THE RECHIT							
				if(trackdirection.z()!=0){				
				// the local position and direction
					Trackhit trackhit_;
					trackhit_.alpha = atan2(trackdirection.z(),trackdirection.x());
					trackhit_.beta = atan2(trackdirection.z(),trackdirection.y());
					trackhit_.gamma = atan2(trackdirection.x(),trackdirection.y());
					trackhit_.x = trackposition.x();
					trackhit_.y = trackposition.y();
					// new instance of rechit to be deleted where TrackLocalAngle::findPixelParameters is called
					SiPixelRecHit* rechitNew = new SiPixelRecHit(*rechit);
					hitangleassociation.push_back(make_pair(rechitNew, trackhit_)); 
				}
			}
			LogDebug("TrackLocalAngle")<<"I found "<<i<<" hits.";
		}
     	delete theTraj;
     	theTraj = 0;
	}
	return hitangleassociation;
}

std::vector<Trajectory> TrackLocalAngle::buildTrajectory(const reco::Track& theT)
{
	TransientTrackingRecHit::RecHitContainer tmp;
	TransientTrackingRecHit::RecHitContainer hits;
	float ndof=0;
  
	for (trackingRecHit_iterator i=theT.recHitsBegin(); i!=theT.recHitsEnd(); i++){
		tmp.push_back(RHBuilder->build(&**i ));
		if ((*i)->isValid()) ndof = ndof + ((*i)->dimension())*((*i)->weight());
	}	

	LogDebug("TrackLocalAngle") << "Transient rechit filled" << "\n";
  
	ndof = ndof - 5;
  
  //SORT RECHITS ALONGMOMENTUM
	const TransientTrackingRecHit::ConstRecHitPointer *firstHit = 0;
	for (TransientTrackingRecHit::RecHitContainer::const_iterator it=tmp.begin(); it!=tmp.end();it++){
		if ((**it).isValid()) {
			firstHit = &(*it);
			break;
		}
	}
	const TransientTrackingRecHit::ConstRecHitPointer *lastHit = 0;
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
  
	TrajectoryStateOnSurface firstState=thePropagator->propagate(theTT.impactPointState(), hits.front()->det()->surface());
	AlgebraicSymMatrix C(5,1);
	C *= 100.;
	TrajectoryStateOnSurface theTSOS( firstState.localParameters(), LocalTrajectoryError(C), firstState.surface(), thePropagator->magneticField()); 
  
	LogDebug("TrackLocalAngle") << "Initial TSOS\n" << theTSOS << "\n";
	PTrajectoryStateOnDet ptsod;
  	const TrajectorySeed seed = TrajectorySeed(PTrajectoryStateOnDet(),BasicTrajectorySeed::recHitContainer(), alongMomentum);
  //perform the fit: the result's size is 1 if it succeded, 0 if fails
	return theFitter->fit(seed, hits, theTSOS);
}

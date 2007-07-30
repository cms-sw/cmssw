
#include <memory>
#include <string>
#include <iostream>
#include <TMath.h>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/TrackLocalAngleTIF.h"

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
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

using namespace std;

TrackLocalAngleTIF::TrackLocalAngleTIF() {
}

TrackLocalAngleTIF::~TrackLocalAngleTIF() {
}

void TrackLocalAngleTIF::init( const edm::EventSetup& es ) {
  //
  // get geometry
  //
  edm::ESHandle<TrackerGeometry> estracker;
  es.get<TrackerDigiGeometryRecord>().get(estracker);
  _tracker=&(* estracker);
}

std::vector<std::pair<const TrackingRecHit*,float> > TrackLocalAngleTIF::SeparateHits(reco::TrackInfoRef & trackinforef) {
  std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;

  // Create the other objects. Use auto_ptr so that ownership is passed when the vector is taken and its memory is correctly freed
  oXZHitAngle = std::auto_ptr<HitAngleAssociation>( new HitAngleAssociation );
  oYZHitAngle = std::auto_ptr<HitAngleAssociation>( new HitAngleAssociation );
             
  oLocalDir = std::auto_ptr<HitLclDirAssociation>( new HitLclDirAssociation );
  oGlobalDir = std::auto_ptr<HitGlbDirAssociation>( new HitGlbDirAssociation );
             
  Hit3DAngle = std::auto_ptr<HitAngleAssociation>( new HitAngleAssociation );

  for(_tkinfoiter=trackinforef->trajStateMap().begin();_tkinfoiter!=trackinforef->trajStateMap().end();++_tkinfoiter) {

    // Check if it is a valid hit
    if (((*_tkinfoiter).first)->isValid()) {

      const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(_tkinfoiter->first)));
      const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(_tkinfoiter->first)));
      const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(_tkinfoiter->first)));
      //     LocalVector trackdirection=(_tkinfoiter->second.parameters()).momentum();
      LocalVector trackdirection=(trackinforef->stateOnDet((*_tkinfoiter).first).parameters()).momentum();

      // Projected Hit
      ////////////////
      if (phit) {
	//phit = POINTER TO THE PROJECTED RECHIT
	hit=&(phit->originalHit());

#ifdef DEBUG
	std::cout << "ProjectedHit found" << std::endl;
#endif

      }

      // Matched Hit
      //////////////
      if(matchedhit){//if matched hit...

#ifdef DEBUG
	std::cout<<"MatchedHit found"<<std::endl;
#endif

	GluedGeomDet * gdet=(GluedGeomDet *)_tracker->idToDet(matchedhit->geographicalId());

	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);

#ifdef DEBUG
	std::cout<<"Track direction trasformed in global direction"<<std::endl;
#endif	

	//cluster and trackdirection on mono det
	
	// THIS THE POINTER TO THE MONO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *monohit=matchedhit->monoHit();
      
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > monocluster=monohit->cluster();
	const GeomDetUnit * monodet=gdet->monoDet();
      
	LocalVector monotkdir=monodet->toLocal(gtrkdir);
	//size=(monocluster->amplitudes()).size();
	// THE LOCAL ANGLE (MONO)
	if(monotkdir.z()!=0) {
	  float angle = atan(monotkdir.x()/ monotkdir.z())*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(monohit, angle));
	  float angleXZ = atan( monotkdir.x()/ monotkdir.z() ); 
	  oXZHitAngle->push_back( std::make_pair( monohit, atan( monotkdir.x()/ monotkdir.z())));
	  oYZHitAngle->push_back( std::make_pair( monohit, atan( monotkdir.y()/ monotkdir.z())));
	  oLocalDir->push_back( std::make_pair( monohit, monotkdir));
	  oGlobalDir->push_back( std::make_pair( monohit, gtrkdir));
	  // 3D angle
	  Hit3DAngle->push_back( std::make_pair( monohit, acos( monotkdir.z()/ monotkdir.mag() ) ) );
	}
	else if ( monotkdir.x() != 0 ) {
	  float angle = ( monotkdir.x()/fabs(monotkdir.z()) )*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(monohit, angle)); 
	  oXZHitAngle->push_back( std::make_pair( monohit, ( monotkdir.x()/fabs(monotkdir.z()) ) ) );
	  oYZHitAngle->push_back( std::make_pair( monohit, ( monotkdir.y()/fabs(monotkdir.z()) ) ) );
	  oLocalDir->push_back( std::make_pair( monohit, monotkdir));
	  oGlobalDir->push_back( std::make_pair( monohit, gtrkdir));
	  // 3D angle
	  Hit3DAngle->push_back( std::make_pair( monohit, acos( monotkdir.z()/ monotkdir.mag() ) ) );
	}
	//std::cout<<"Angle="<<atan(monotkdir.x(), monotkdir.z())*180/TMath::Pi()<<std::endl;
	
	//cluster and trackdirection on stereo det
	
	// THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *stereohit=matchedhit->stereoHit();
	
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
	const GeomDetUnit * stereodet=gdet->stereoDet(); 
	LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	//size=(stereocluster->amplitudes()).size();
	// THE LOCAL ANGLE (STEREO)
	if(stereotkdir.z()!=0){
	  float angle = atan(stereotkdir.x()/ stereotkdir.z())*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(stereohit, angle)); 
	  oXZHitAngle->push_back( std::make_pair( stereohit, atan( stereotkdir.x()/ stereotkdir.z())));
	  oYZHitAngle->push_back( std::make_pair( stereohit, atan( stereotkdir.y()/ stereotkdir.z())));
	  oLocalDir->push_back( std::make_pair( stereohit, stereotkdir));
	  oGlobalDir->push_back( std::make_pair( stereohit, gtrkdir));
	  // 3D angle
	  Hit3DAngle->push_back( std::make_pair( stereohit, acos( stereotkdir.z()/ stereotkdir.mag() ) ) );
	}
	else if ( stereotkdir.x() != 0 ) {
	  float angle = ( stereotkdir.x()/fabs(stereotkdir.z()) )*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(stereohit, angle)); 
	  oXZHitAngle->push_back( std::make_pair( stereohit, ( stereotkdir.x()/fabs(stereotkdir.z()) ) ) );
	  oYZHitAngle->push_back( std::make_pair( stereohit, ( stereotkdir.y()/fabs(stereotkdir.z()) ) ) );
	  oLocalDir->push_back( std::make_pair( stereohit, stereotkdir));
	  oGlobalDir->push_back( std::make_pair( stereohit, gtrkdir));
	  // 3D angle
	  Hit3DAngle->push_back( std::make_pair( stereohit, acos( stereotkdir.z()/ stereotkdir.mag() ) ) );
	}
      }

      // Single sided detector Hit
      ////////////////////////////
      else if( hit ) {
	//  hit= POINTER TO THE RECHIT
	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
	GeomDet * gdet=(GeomDet *)_tracker->idToDet(hit->geographicalId());
	//size=(cluster->amplitudes()).size();
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
      
	// THE LOCAL ANGLE (STEREO)
	if(trackdirection.z()!=0){
	  float angle = atan(trackdirection.x()/ trackdirection.z())*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(hit, angle)); 
	  oXZHitAngle->push_back( std::make_pair( hit, atan( trackdirection.x()/ trackdirection.z())));
	  oYZHitAngle->push_back( std::make_pair( hit, atan( trackdirection.y()/ trackdirection.z())));
	  oLocalDir->push_back( std::make_pair( hit, trackdirection));
	  oGlobalDir->push_back( std::make_pair( hit, gtrkdir));
	  // 3D angle
	  Hit3DAngle->push_back( std::make_pair( hit, acos( trackdirection.z()/ trackdirection.mag() ) ) );
	}
	else if ( trackdirection.x() != 0 ) {
	  float angle = ( trackdirection.x()/fabs(trackdirection.z()) )*180/TMath::Pi();
	  hitangleassociation.push_back(std::make_pair(hit, angle)); 
	  oXZHitAngle->push_back( std::make_pair( hit, ( trackdirection.x()/fabs(trackdirection.z()) ) ) );
	  oYZHitAngle->push_back( std::make_pair( hit, ( trackdirection.y()/fabs(trackdirection.z()) ) ) );
	  oLocalDir->push_back( std::make_pair( hit, trackdirection));
	  oGlobalDir->push_back( std::make_pair( hit, gtrkdir));
	  // 3D angle
	  Hit3DAngle->push_back( std::make_pair( hit, acos( trackdirection.z()/ trackdirection.mag() ) ) );
	}
      }
      else {
#ifdef DEBUG
	std::cout << "not matched, mono or projected rechit" << std::endl;
#endif
      }
    } // end if is valid hit
  } // end loop on rechits
  return (hitangleassociation);
}

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

vector<pair<const TrackingRecHit*,float> > TrackLocalAngleTIF::SeparateHits(reco::TrackInfoRef & trackinforef) {
  std::vector<std::pair<const TrackingRecHit*,float> >hitangleassociation;
  for(_tkinfoiter=trackinforef->trajStateMap().begin();_tkinfoiter!=trackinforef->trajStateMap().end();++_tkinfoiter){
    const ProjectedSiStripRecHit2D* phit=dynamic_cast<const ProjectedSiStripRecHit2D*>(&(*(_tkinfoiter->first)));
    const SiStripMatchedRecHit2D* matchedhit=dynamic_cast<const SiStripMatchedRecHit2D*>(&(*(_tkinfoiter->first)));
    const SiStripRecHit2D* hit=dynamic_cast<const SiStripRecHit2D*>(&(*(_tkinfoiter->first)));
    LocalVector trackdirection=(_tkinfoiter->second.parameters()).momentum();
    if (phit) {//if projected hit...
      hit = &(phit->originalHit());
    }
    if(matchedhit){//if matched hit...
	
      GluedGeomDet * gdet=(GluedGeomDet *)_tracker->idToDet(matchedhit->geographicalId());
	
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
	float angle = atan2(monotkdir.x(), monotkdir.z())*180/TMath::Pi();
	//
	hitangleassociation.push_back(make_pair(monohit, angle)); 
	oXZHitAngle.push_back( make_pair( monohit, atan2( monotkdir.x(), monotkdir.z())));
	oYZHitAngle.push_back( make_pair( monohit, atan2( monotkdir.y(), monotkdir.z())));
	oLocalDir.push_back( make_pair( monohit, monotkdir));
	oGlobalDir.push_back( make_pair( monohit, gtrkdir));
	//	  std::cout<<"Angle="<<atan2(monotkdir.x(), monotkdir.z())*180/TMath::Pi()<<std::endl;

	//cluster and trackdirection on stereo det
	    
	// THIS THE POINTER TO THE STEREO HIT OF A MATCHED HIT 
	const SiStripRecHit2D *stereohit=matchedhit->stereoHit();

	const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > stereocluster=stereohit->cluster();
	const GeomDetUnit * stereodet=gdet->stereoDet(); 
	LocalVector stereotkdir=stereodet->toLocal(gtrkdir);
	//size=(stereocluster->amplitudes()).size();
	if(stereotkdir.z()!=0){
	    
	  // THE LOCAL ANGLE (STEREO)
	  float angle = atan2(stereotkdir.x(), stereotkdir.z())*180/TMath::Pi();
	  hitangleassociation.push_back(make_pair(stereohit, angle)); 
	  oXZHitAngle.push_back( make_pair( stereohit, atan2( stereotkdir.x(), stereotkdir.z())));
	  oYZHitAngle.push_back( make_pair( stereohit, atan2( stereotkdir.y(), stereotkdir.z())));
	  oLocalDir.push_back( make_pair( stereohit, stereotkdir));
	  oGlobalDir.push_back( make_pair( stereohit, gtrkdir));
	}
      }
    }
    else if(hit){
      //  hit= POINTER TO THE RECHIT
      const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > cluster=hit->cluster();
      GeomDet * gdet=(GeomDet *)_tracker->idToDet(hit->geographicalId());
      //size=(cluster->amplitudes()).size();

      if(trackdirection.z()!=0){

	// THE LOCAL ANGLE (STEREO)
	float angle = atan2(trackdirection.x(), trackdirection.z())*180/TMath::Pi();
	hitangleassociation.push_back(make_pair(hit, angle)); 
	oXZHitAngle.push_back( make_pair( hit, atan2( trackdirection.x(), trackdirection.z())));
	oYZHitAngle.push_back( make_pair( hit, atan2( trackdirection.y(), trackdirection.z())));
	oLocalDir.push_back( make_pair( hit, trackdirection));
	GlobalVector gtrkdir=gdet->toGlobal(trackdirection);
	oGlobalDir.push_back( make_pair( hit, gtrkdir));
      }
    }
    else {
      std::cout << "not matched nor mono, maybe it is ProjectedRecHit" << std::endl;
    }
  } // end loop on rechits
  return (hitangleassociation);
}



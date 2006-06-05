// File: SiStripDetHitConverterAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHitAlgorithm.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

ReadRecHitAlgorithm::ReadRecHitAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

ReadRecHitAlgorithm::~ReadRecHitAlgorithm() {
}


void ReadRecHitAlgorithm::run(const SiStripRecHit2DLocalPosCollection* input)
{
  
  // get vector of detunit ids
  const std::vector<DetId> detIDs = input->ids();
  //  std::cout<<detIDs.size()<<std::endl;
    // loop over detunits
  for ( std::vector<DetId>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    unsigned int id = (*detunit_iterator).rawId();
    edm::OwnVector<SiStripRecHit2DLocalPos> collector; 
    if(id!=999999999){ //if is valid detector
      SiStripRecHit2DLocalPosCollection::range rechitRange = input->get((*detunit_iterator));
      SiStripRecHit2DLocalPosCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripRecHit2DLocalPosCollection::const_iterator rechitRangeIteratorEnd   = rechitRange.second;
      SiStripRecHit2DLocalPosCollection::const_iterator iter=rechitRangeIteratorBegin;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripRecHit2DLocalPos const rechit=*iter;
	  LocalPoint position=rechit.localPosition();
	  LocalError error=rechit.localPositionError();
	  //GeomDet& det=rechit->det();
	  //DetId id=rechit.geographicalId();
	  const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=rechit.cluster();
	  edm::LogInfo("ReadRecHit")<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<"\n"
	  <<"local error: "<<error.xx()<<" "<<error.xy()<<" "<<error.yy();
	  if (clust.isNonnull ()){
	    edm::LogInfo("ReadRecHit")<<"barycenter= "<<clust->barycenter();
	  }
	  else{
	    edm::LogError("ReadRecHit")<<"The cluster is empty!";
	  }
	  //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
    }
  }
};


void ReadRecHitAlgorithm::run(const SiStripRecHit2DMatchedLocalPosCollection* input)
{
  
  // get vector of detunit ids
  const std::vector<DetId> detIDs = input->ids();
  //  std::cout<<detIDs.size()<<std::endl;
    // loop over detunits
  for ( std::vector<DetId>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    unsigned int id = (*detunit_iterator).rawId();
    edm::OwnVector<SiStripRecHit2DLocalPos> collector; 
    if(id!=999999999){ //if is valid detector
      SiStripRecHit2DMatchedLocalPosCollection::range rechitRange = input->get((*detunit_iterator));
      SiStripRecHit2DMatchedLocalPosCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripRecHit2DMatchedLocalPosCollection::const_iterator rechitRangeIteratorEnd   = rechitRange.second;
      SiStripRecHit2DMatchedLocalPosCollection::const_iterator iter=rechitRangeIteratorBegin;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripRecHit2DMatchedLocalPos const rechit=*iter;
	  LocalPoint position=rechit.localPosition();
	  LocalError error=rechit.localPositionError();
	  //GeomDet& det=rechit->det();
	  //DetId id=rechit.geographicalId();
	  //	  std::vector<const SiStripCluster*> clust=rechit.cluster();
	  edm::LogInfo("ReadRecHit")<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<"\n"
	  <<"local error: "<<error.xx()<<" "<<error.xy()<<" "<<error.yy();
	  //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
    }
  }
}




  

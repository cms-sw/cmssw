// File: ReadRecHitAlgorithm.cc
// Description:  Analyzer that reads rechits
// Author:  C. Genta
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHitAlgorithm.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

ReadRecHitAlgorithm::ReadRecHitAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

ReadRecHitAlgorithm::~ReadRecHitAlgorithm() {
}


void ReadRecHitAlgorithm::run(const SiStripRecHit2DCollection* input)
{
  
  // get vector of detunit ids
  const std::vector<DetId> detIDs = input->ids();
  //  std::cout<<detIDs.size()<<std::endl;
    // loop over detunits
  for ( std::vector<DetId>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    unsigned int id = (*detunit_iterator).rawId();
    edm::OwnVector<SiStripRecHit2D> collector; 
    if(id!=999999999){ //if is valid detector
      SiStripRecHit2DCollection::range rechitRange = input->get((*detunit_iterator));
      SiStripRecHit2DCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripRecHit2DCollection::const_iterator rechitRangeIteratorEnd   = rechitRange.second;
      SiStripRecHit2DCollection::const_iterator iter=rechitRangeIteratorBegin;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripRecHit2D const rechit=*iter;
	  LocalPoint position=rechit.localPosition();
	  LocalError error=rechit.localPositionError();
	  //GeomDet& det=rechit->det();
	  //DetId id=rechit.geographicalId();
	  SiStripRecHit2D::ClusterRef clust=rechit.cluster();
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
}


void ReadRecHitAlgorithm::run(const SiStripMatchedRecHit2DCollection* input)
{
  
  // get vector of detunit ids
  const std::vector<DetId> detIDs = input->ids();
  //  std::cout<<detIDs.size()<<std::endl;
    // loop over detunits
  for ( std::vector<DetId>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    unsigned int id = (*detunit_iterator).rawId();
    edm::OwnVector<SiStripRecHit2D> collector; 
    if(id!=999999999){ //if is valid detector
      SiStripMatchedRecHit2DCollection::range rechitRange = input->get((*detunit_iterator));
      SiStripMatchedRecHit2DCollection::const_iterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripMatchedRecHit2DCollection::const_iterator rechitRangeIteratorEnd   = rechitRange.second;
      SiStripMatchedRecHit2DCollection::const_iterator iter=rechitRangeIteratorBegin;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripMatchedRecHit2D const rechit=*iter;
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




  

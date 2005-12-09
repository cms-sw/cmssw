// File: SiStripDetHitConverterAlgorithm.cc
// Description:  An algorithm for CMS track reconstruction.
// Author:  O/ Gutsche
// Creation Date:  OGU Aug. 1, 2005   

#include <vector>
#include <algorithm>
#include <iostream>

#include "RecoLocalTracker/SiStripRecHitConverter/test/ReadRecHitAlgorithm.h"
#include "PhysicsTools/Candidate/interface/own_vector.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include "Geometry/TrackerSimAlgo/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/CommonDetAlgo/interface/MeasurementPoint.h"

#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

using namespace std;

ReadRecHitAlgorithm::ReadRecHitAlgorithm(const edm::ParameterSet& conf) : conf_(conf) { 
}

ReadRecHitAlgorithm::~ReadRecHitAlgorithm() {
}


void ReadRecHitAlgorithm::run(const SiStripRecHit2DLocalPosCollection* input)
{
  
  // get vector of detunit ids
  const std::vector<unsigned int> detIDs = input->detIDs();
  //  std::cout<<detIDs.size()<<std::endl;
    // loop over detunits
  for ( std::vector<unsigned int>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    unsigned int id = *detunit_iterator;
    own_vector<SiStripRecHit2DLocalPos> collector; 
    if(id!=999999999){ //if is valid detector
       const SiStripRecHit2DLocalPosCollection::Range rechitRange = input->get(id);
      SiStripRecHit2DLocalPosCollection::ContainerIterator rechitRangeIteratorBegin = rechitRange.first;
      SiStripRecHit2DLocalPosCollection::ContainerIterator rechitRangeIteratorEnd   = rechitRange.second;
      SiStripRecHit2DLocalPosCollection::ContainerIterator iter;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripRecHit2DLocalPos * const rechit=*iter;
	  LocalPoint position=rechit->localPosition();
	  LocalError error=rechit->localPositionError();
	  //GeomDet& det=rechit->det();
	  DetId id=rechit->geographicalId();
	  const SiStripCluster* clust=rechit->cluster();
	  std::cout<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<" "<<std::endl;
	  //std::cout<<"local error: "<<error.x()<<" "<<error.y()<<" "<<error.z()<<" "<<std::endl;
	  //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
    }
  }
};
  

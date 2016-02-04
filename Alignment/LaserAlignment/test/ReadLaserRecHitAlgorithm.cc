/** \file ReadLaserRecHitAlgorithm.cc
 *  Algorithm to read RecHits from the LaserBeams
 *
 *  $Date: 2008/11/07 11:04:19 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/test/ReadLaserRecHitAlgorithm.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/Framework/interface/EventSetup.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h" 
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h" 
#include "DataFormats/GeometrySurface/interface/LocalError.h" 
#include "DataFormats/GeometryVector/interface/LocalPoint.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h" 
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h" 

ReadLaserRecHitAlgorithm::ReadLaserRecHitAlgorithm(const edm::ParameterSet& conf) : conf_(conf) {}

ReadLaserRecHitAlgorithm::~ReadLaserRecHitAlgorithm() {}


void ReadLaserRecHitAlgorithm::run(const SiStripRecHit2DCollection* input, const edm::EventSetup& theSetup)
{
  // access the Tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  const TrackerGeometry& theTracker(*theTrackerGeometry);

  for ( SiStripRecHit2DCollection::const_iterator detunit_iterator = input->begin(), detunit_end = input->end(); 
            detunit_iterator != detunit_end; ++detunit_iterator) 
  {
    SiStripRecHit2DCollection::DetSet rechitRange = *detunit_iterator;
    unsigned int id = detunit_iterator->detId();

    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet(DetId(id)));

    if(id!=999999999){ //if is valid detector
      SiStripRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = rechitRange.begin();
      SiStripRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd   = rechitRange.end();
      SiStripRecHit2DCollection::DetSet::const_iterator iter=rechitRangeIteratorBegin;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripRecHit2D const rechit=*iter;
	  LocalPoint position=rechit.localPosition();
	  LocalError error=rechit.localPositionError();
	  GlobalPoint gPosition = theStripDet->surface().toGlobal(position);

	  ErrorFrameTransformer theErrorTransformer;
	  GlobalError gError = theErrorTransformer.transform(error, theStripDet->surface());


	  //GeomDet& det=rechit->det();
	  //DetId id=rechit.geographicalId();
//  	  std::vector<const SiStripCluster*> clust=rechit.cluster();
	  std::cout << "local position:\t" << position.x() << "\t" << position.y() << "\t" << position.z() << "\t\t"
		    << "global position:\t" << gPosition.x() << "\t" << gPosition.y() << "\t" << gPosition.z() << std::endl;
	  std::cout << "local error:\t" << error.xx() << "\t" << error.xy() << "\t" <<error.yy() << "\t\t" 
		    << "global error:\t" << gError.cxx() << "\t" << gError.cyy() << "\t" << gError.czz() << std::endl;
	  //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
    }
  }
}


void ReadLaserRecHitAlgorithm::run(const SiStripMatchedRecHit2DCollection* input, const edm::EventSetup& theSetup)
{
    // loop over detunits
  for ( SiStripMatchedRecHit2DCollection::const_iterator detunit_iterator = input->begin(), detunit_end = input->end(); 
            detunit_iterator != detunit_end; ++detunit_iterator) 
  {
    SiStripMatchedRecHit2DCollection::DetSet rechitRange = *detunit_iterator;
    unsigned int id = detunit_iterator->detId();
    if(id!=999999999){ //if is valid detector
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorBegin = rechitRange.begin();
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitRangeIteratorEnd   = rechitRange.end();
      SiStripMatchedRecHit2DCollection::DetSet::const_iterator iter=rechitRangeIteratorBegin;
      for(iter=rechitRangeIteratorBegin;iter!=rechitRangeIteratorEnd;++iter){//loop on the rechit
	  SiStripMatchedRecHit2D const rechit=*iter;
	  LocalPoint position=rechit.localPosition();
	  LocalError error=rechit.localPositionError();
	  //GeomDet& det=rechit->det();
	  //DetId id=rechit.geographicalId();
	  //	  std::vector<const SiStripCluster*> clust=rechit.cluster();
	  std::cout<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<" "<<std::endl;
	  std::cout<<"local error: "<<error.xx()<<" "<<error.xy()<<" "<<error.yy()<<" "<<std::endl;
	  //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
    }
  }
}




  

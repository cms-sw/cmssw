/*
 * Algorith to read RecHits from the LaserBeams
 */

#include "Alignment/LaserAlignment/test/ReadLaserRecHitAlgorithm.h"

ReadLaserRecHitAlgorithm::ReadLaserRecHitAlgorithm(const edm::ParameterSet& conf) : conf_(conf) {}

ReadLaserRecHitAlgorithm::~ReadLaserRecHitAlgorithm() {}


void ReadLaserRecHitAlgorithm::run(const SiStripRecHit2DCollection* input, const edm::EventSetup& theSetup)
{
  // access the Tracker
  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  theSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  const TrackerGeometry& theTracker(*theTrackerGeometry);

  // get vector of detunit ids
  const std::vector<DetId> detIDs = input->ids();
  //  std::cout<<detIDs.size()<<std::endl;
    // loop over detunits
  for ( std::vector<DetId>::const_iterator detunit_iterator = detIDs.begin(); detunit_iterator != detIDs.end(); detunit_iterator++ ) {//loop over detectors
    unsigned int id = (*detunit_iterator).rawId();

    const StripGeomDetUnit* const theStripDet = dynamic_cast<const StripGeomDetUnit*>(theTracker.idToDet((*detunit_iterator)));

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
	  std::cout<<"local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<" "<<std::endl;
	  std::cout<<"local error: "<<error.xx()<<" "<<error.xy()<<" "<<error.yy()<<" "<<std::endl;
	  //	  std::cout<<"det id: "<<id.rawid<<std::endl;
      }
    }
  }
}




  

#include <memory>

// Framework
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
//
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// PSimHits
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
// RecHits
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/Common/interface/OwnVector.h"
// std
#include <iostream>
#include <string>
//
#include "FastSimulation/TrackingRecHitProducer/test/FamosRecHitTest.h"
//

FamosRecHitTest::FamosRecHitTest(edm::ParameterSet const& pset)
{
  std::cout << "Start Famos RecHit Test" << std::endl;
}

void FamosRecHitTest::beginJob() {}

void FamosRecHitTest::endJob() {}

// Virtual destructor needed.
FamosRecHitTest::~FamosRecHitTest() {
  std::cout << "End Famos RecHit Test" << std::endl;
}

void FamosRecHitTest::analyze(const edm::Event& event, const edm::EventSetup& setup)
{
  // get event and run number
  int t_Run   = event.id().run();
  int t_Event = event.id().event();
  std::cout
    << " #################################### Run " << t_Run
    << " Event "                                    << t_Event << " #################################### "
    << std::endl;

 

  edm::Handle<SiTrackerGSRecHit2DCollection> theRecHits;
  event.getByLabel("siTrackerGaussianSmearingRecHits", "TrackerGSRecHits", theRecHits);
  std::cout << "Size of Rechit collection\t" << theRecHits->size() << std::endl;
 
 
 edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theRecHitsMatched;
 event.getByLabel("siTrackerGaussianSmearingRecHits", "TrackerGSMatchedRecHits", theRecHitsMatched);
 std::cout << "Size of Rechit Matched collection\t" << theRecHitsMatched->size() << std::endl;
 
 const std::vector<unsigned> theSimTrackIds = theRecHitsMatched->ids();
 
 for ( unsigned tkId=0;  tkId != theSimTrackIds.size(); ++tkId ) {
   std::cout << "Track number " << tkId << std::endl;
   unsigned simTrackId = theSimTrackIds[tkId];
   SiTrackerGSMatchedRecHit2DCollection::range theRecHitRange = theRecHitsMatched->get(simTrackId);
   SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorBegin = theRecHitRange.first;
   SiTrackerGSMatchedRecHit2DCollection::const_iterator theRecHitRangeIteratorEnd   = theRecHitRange.second;
   SiTrackerGSMatchedRecHit2DCollection::const_iterator iterRecHit;
   
   for ( iterRecHit = theRecHitRangeIteratorBegin; 
	 iterRecHit != theRecHitRangeIteratorEnd; 
	 ++iterRecHit) { 
     
     const SiTrackerGSMatchedRecHit2D* rechit= &(*iterRecHit);
     LocalPoint position=rechit->localPosition();
     std::cout <<"Matched: local position: "<<position.x()<<" "<<position.y()<<" "<<position.z()<<"\n"
	       << "Is Matched = " << rechit->isMatched()<< std::endl;
     // if(rechit->isMatched()){
       
       const SiTrackerGSRecHit2D* rphihit = rechit->monoHit();
       LocalPoint position_phi=rphihit->localPosition();
       std::cout <<"Phi Component: local position: "<<position_phi.x()<<" "<<position_phi.y()<<" "<<position_phi.z()<< std::endl;
       
       if(!rechit->stereoHit()) continue;
       const SiTrackerGSRecHit2D* sashit = rechit->stereoHit();
       LocalPoint position_sas=sashit->localPosition();
       std::cout <<"Stereo Component: local position: "<<position_sas.x()<<" "<<position_sas.y()<<" "<<position_sas.z()<< std::endl;
       
       // }
   }
 }
}

DEFINE_FWK_MODULE(FamosRecHitTest);

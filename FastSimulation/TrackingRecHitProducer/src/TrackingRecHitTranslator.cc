/** TrackingRecHitTranslator.cc
 * --------------------------------------------------------------
 * Description:  see TrackingRecHitTranslator.h
 * Authors:  R. Ranieri (CERN)
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */


// SiTracker Gaussian Smearing
#include "TrackingRecHitTranslator.h"

// Geometry
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// Data Formats
#include "DataFormats/Common/interface/Handle.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data Formats
#include "DataFormats/DetId/interface/DetId.h"

// STL
#include <memory>
#include <string>

TrackingRecHitTranslator::TrackingRecHitTranslator(edm::ParameterSet const& conf) :
  hitCollectionInputTag_(conf.getParameter<edm::InputTag>("hitCollectionInputTag"))
{
  produces<SiTrackerFullGSRecHit2DCollection>();
}

// Destructor
TrackingRecHitTranslator::~TrackingRecHitTranslator() {}  

void 
TrackingRecHitTranslator::beginRun(edm::Run const&, const edm::EventSetup & es) {

  // Initialize the Tracker Geometry
  edm::ESHandle<TrackerGeometry> theGeometry;
  es.get<TrackerDigiGeometryRecord> ().get (theGeometry);
  geometry = &(*theGeometry);

}

void TrackingRecHitTranslator::produce(edm::Event& e, const edm::EventSetup& es) 
{
  // Step A: Get Inputs (FastGSRecHit's)
  edm::Handle<SiTrackerGSRecHit2DCollection> theFastRecHits; 
  e.getByLabel(hitCollectionInputTag_, theFastRecHits);

  // Step B: fill a temporary full RecHit collection from the fast RecHit collection
  SiTrackerGSRecHit2DCollection::const_iterator aHit = theFastRecHits->begin();
  SiTrackerGSRecHit2DCollection::const_iterator theLastHit = theFastRecHits->end();
  std::map< DetId, edm::OwnVector<SiTrackerGSRecHit2D> > temporaryRecHits;
    
  // loop on Fast GS Hits
  for ( ; aHit != theLastHit; ++aHit ) {

    DetId det = aHit->geographicalId();

    /* 
    const GeomDet* theDet = geometry->idToDet(det);
    unsigned trackID = aHit->simtrackId();

    std::cout << "Track/z/r after : "
	      << trackID << " " 
	      << theDet->surface().toGlobal(aHit->localPosition()).z() << " " 
	      << theDet->surface().toGlobal(aHit->localPosition()).perp() << std::endl;
    */

    // create RecHit
    // Fill the temporary RecHit on the current DetId collection
    temporaryRecHits[det].push_back(aHit->clone());

  }

  // Step C: from the temporary RecHit collection, create the real one.
  std::auto_ptr<SiTrackerFullGSRecHit2DCollection> 
    recHitCollection(new SiTrackerFullGSRecHit2DCollection);
  loadRecHits(temporaryRecHits, *recHitCollection);
  
  // Step D: write output to file
  e.put(recHitCollection);

}

void 
TrackingRecHitTranslator::loadRecHits(
     std::map<DetId,edm::OwnVector<SiTrackerGSRecHit2D> >& theRecHits, 
     SiTrackerFullGSRecHit2DCollection& theRecHitCollection) const
{
  std::map<DetId,edm::OwnVector<SiTrackerGSRecHit2D> >::const_iterator 
    it = theRecHits.begin();
  std::map<DetId,edm::OwnVector<SiTrackerGSRecHit2D> >::const_iterator 
    lastRecHit = theRecHits.end();

  for( ; it != lastRecHit ; ++it ) { 
    theRecHitCollection.put(it->first,it->second.begin(),it->second.end());
  }

}


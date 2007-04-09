/** TrackingRecHitTranslator.cc
 * --------------------------------------------------------------
 * Description:  see TrackingRecHitTranslator.h
 * Authors:  R. Ranieri (CERN)
 * History: Sep 27, 2006 -  initial version
 * --------------------------------------------------------------
 */


// SiTracker Gaussian Smearing
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitTranslator.h"

// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

// Data Formats
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"

// Framework
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

// Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

// Random engine
#include "FastSimulation/Utilities/interface/RandomEngine.h"

// STL
#include <memory>
#include <string>
#include <iostream>

TrackingRecHitTranslator::TrackingRecHitTranslator(
  edm::ParameterSet const& conf) 
  : conf_(conf)
{
  produces<SiTrackerFullGSRecHit2DCollection>();
}

// Destructor
TrackingRecHitTranslator::~TrackingRecHitTranslator() {}  

void TrackingRecHitTranslator::beginJob(const edm::EventSetup& es) {

  // Initialize the Tracker Geometry
  edm::ESHandle<TrackerGeometry> theGeometry;
  es.get<TrackerDigiGeometryRecord> ().get (theGeometry);
  geometry = &(*theGeometry);

}

void TrackingRecHitTranslator::produce(edm::Event& e, const edm::EventSetup& es) 
{
  // Step A: Get Inputs (FastGSRecHit's)
  edm::Handle<SiTrackerGSRecHit2DCollection> theFastRecHits; 
  e.getByType(theFastRecHits);

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


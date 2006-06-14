
/*
 *  See header file for a description of this class.
 *
 *  $Date: $
 *  $Revision: $
 *  \author M. Giunta
 */

#include "DTVDriftCalibration.h"

#include "RecoLocalMuon/DTSegment/test/DTRecSegment4DReader.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"


using namespace std;
using namespace edm;


DTVDriftCalibration::DTVDriftCalibration(const ParameterSet& pset) {
  debug = pset.getUntrackedParameter<bool>("debug","false");

  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getUntrackedParameter<string>("recHits4DLabel");

}

DTVDriftCalibration::~DTVDriftCalibration(){}



void DTVDriftCalibration::analyze(const Event & event, const EventSetup& eventSetup) {
  cout << endl<<"--- [DTVDriftCalibration] Event analysed #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;
   // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the rechit collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // Loop over all 4D segments in the collection
  for(DTRecSegment4DCollection::const_iterator segment = all4DSegments->begin();
	segment != all4DSegments->end();
	++segment){
    cout << " Segment: " << *segment <<endl;
  }

  // Loop over segments by chamber
  DTRecSegment4DCollection::id_iterator chamberIdIt;
  for (chamberIdIt = all4DSegments->id_begin();
       chamberIdIt != all4DSegments->id_end();
       ++chamberIdIt){
    // Get the chamber from the setup
    const DTChamber* chamber = dtGeom->chamber(*chamberIdIt);
      cout << "Chamber Id: " << *chamberIdIt << endl;
    // Get the range for the corresponding LayerId
    DTRecSegment4DCollection::range  range = all4DSegments->get((*chamberIdIt));
    // Loop over the rechits of this DetUnit
    for (DTRecSegment4DCollection::const_iterator segment = range.first;
	 segment!=range.second;
	 ++segment){
      cout << "Segment local pos (in chamber RF): " << (*segment).localPosition() << endl;
      cout << "Segment global pos: " << chamber->toGlobal((*segment).localPosition()) << endl;;
    } 
  }
}


void DTVDriftCalibration::endJob() {
  cout << "End of job" << endl;
  // Here fit the histos, write them to file and write DB objects to DB
  
  
}





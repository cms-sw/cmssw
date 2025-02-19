/** \class DTSegment4DT0Corrector
 *  Builds the segments in the DT chambers.
 *
 *  $Date: 2009/03/10 16:09:13 $
 *  $Revision: 1.2 $
 * \author Mario Pelliccioni - INFN Torino <pellicci@cern.ch>
 */

#include "RecoLocalMuon/DTSegment/src/DTSegment4DT0Corrector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace edm;
using namespace std;

DTSegment4DT0Corrector::DTSegment4DT0Corrector(const ParameterSet& pset){
  produces<DTRecSegment4DCollection>();
  
  // debug parameter
  debug = pset.getUntrackedParameter<bool>("debug"); 
  
  if(debug)
    cout << "[DTSegment4DT0Corrector] Constructor called" << endl;
  
  // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<InputTag>("recHits4DLabel");

    // the updator
    theUpdator = new DTSegmentUpdator(pset);
}

/// Destructor
DTSegment4DT0Corrector::~DTSegment4DT0Corrector(){
  if(debug)
    cout << "[DTSegment4DT0Corrector] Destructor called" << endl;
  delete theUpdator;
}

void DTSegment4DT0Corrector::produce(Event& event, const EventSetup& setup){

  // Get the 4D Segment from the event
  Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  // get the geometry
  ESHandle<DTGeometry> theGeom;
  setup.get<MuonGeometryRecord>().get(theGeom);

 // Percolate the setup
  theUpdator->setES(setup);


  // Create the pointer to the collection which will store the rechits
  auto_ptr<DTRecSegment4DCollection> segments4DCollection(new DTRecSegment4DCollection());

  // Iterate over the input DTSegment4D
  DTRecSegment4DCollection::id_iterator chamberId;

  if(debug)
    cout << "[DTSegment4DT0Corrector] Starting to loop over segments" << endl;

  for (chamberId = all4DSegments->id_begin(); chamberId != all4DSegments->id_end(); ++chamberId){

    OwnVector<DTRecSegment4D> result;

    // Get the range for the corresponding ChamerId
    DTRecSegment4DCollection::range  range = all4DSegments->get(*chamberId);

    // Loop over the rechits of this ChamberId
    for (DTRecSegment4DCollection::const_iterator segment4D = range.first;
	 segment4D!=range.second; ++segment4D) {

      DTRecSegment4D tmpseg = *segment4D;

      DTRecSegment4D *newSeg = tmpseg.clone();

      if(newSeg == 0) continue;

      theUpdator->update(newSeg,true);
      result.push_back(*newSeg);

    }

    segments4DCollection->put(*chamberId, result.begin(), result.end());

  }

  if(debug)
    cout << "[DTSegment4DT0Corrector] Saving modified segments into the event" << endl;

  // Load the output in the Event
  event.put(segments4DCollection);
}

/** \class DTRecSegment4DProducer
 *  Builds the segments in the DT chambers.
 *
 *  $Date: 2006/04/21 14:25:38 $
 *  $Revision: 1.2 $
 * \author Riccardo Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DProducer.h"

#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DPhi.h"
#include "DataFormats/Common/interface/OwnVector.h"

#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DAlgoFactory.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
using namespace edm;
using namespace std;

DTRecSegment4DProducer::DTRecSegment4DProducer(const ParameterSet& pset){
  produces<DTRecSegment4DCollection>();
  
  // debug parameter
  debug = pset.getUntrackedParameter<bool>("debug"); 
  
  if(debug)
    cout << "[DTRecSegment4DProducer] Constructor called" << endl;
  
  // the name of the 1D rec hits collection
  theRecHits1DLabel = pset.getParameter<string>("recHits1DLabel");
  
  // the name of the 2D rec hits collection
  theRecHits2DLabel = pset.getParameter<string>("recHits2DLabel");
  
  // Get the concrete 4D-segments reconstruction algo from the factory
  string theReco4DAlgoName = pset.getParameter<string>("Reco4DAlgoName");
  cout << "the Reco4D AlgoName is " << theReco4DAlgoName << endl;
  the4DAlgo = DTRecSegment4DAlgoFactory::get()->create(theReco4DAlgoName,
						       pset.getParameter<ParameterSet>("Reco4DAlgoConfig"));
}

/// Destructor
DTRecSegment4DProducer::~DTRecSegment4DProducer(){
  if(debug)
    cout << "[DTRecSegment4DProducer] Destructor called" << endl;
}

void DTRecSegment4DProducer::produce(Event& event, const EventSetup& setup){

  // Get the 1D rechits from the event
  Handle<DTRecHitCollection> all1DHits; 
  event.getByLabel(theRecHits1DLabel,"DT1DRecHits",all1DHits);
  
  // Get the 2D rechits from the event
  Handle<DTRecSegment2DCollection> all2DSegments; 
  event.getByLabel(theRecHits2DLabel, all2DSegments);

  // Create the pointer to the collection which will store the rechits
  auto_ptr<DTRecSegment4DCollection> segments4DCollection(new DTRecSegment4DCollection());

  // get the geometry
  ESHandle<DTGeometry> theGeom;
  setup.get<MuonGeometryRecord>().get(theGeom);

  // Percolate the setup
  the4DAlgo->setES(setup);

  // Iterate over all hit collections ordered by SuperLayerId
  DTRecSegment2DCollection::id_iterator dtSuperLayerIt;
  DTChamberId oldChId;

  for (dtSuperLayerIt = all2DSegments->id_begin(); dtSuperLayerIt != all2DSegments->id_end(); ++dtSuperLayerIt){

    // The superLayerId
    DTSuperLayerId superLayerId = (*dtSuperLayerIt);

    // Check the DTChamberId
    const DTChamberId chId = superLayerId.chamberId();
    if (chId==oldChId) continue; // I'm on the same Chamber as before
    oldChId = chId;
    if(debug) cout << "ChamberId: "<< chId << endl;
    the4DAlgo->setChamber(chId);

    cout<<"Take the DTRecHits1D in the Phi-SLs and set them in the reconstructor"<<endl;

    the4DAlgo->setDTRecHit1DContainer(all1DHits);

    cout<<"Take the DTRecSegments2D and set them in the reconstructor"<<endl;

    the4DAlgo->setDTRecSegment2DContainer(all2DSegments);

    cout << "Start 4D-Segments Reco " << endl;
    
    OwnVector<DTRecSegment4D> segments4D = the4DAlgo->reconstruct();
    
    cout << "Number of reconstructed 4D-segments " << segments4D.size() << endl;

    if (segments4D.size() > 0 )
      // convert the OwnVector into a Collection
      segments4DCollection->put(chId, segments4D.begin(),segments4D.end());
  }
  // Load the output in the Event
  event.put(segments4DCollection);
}

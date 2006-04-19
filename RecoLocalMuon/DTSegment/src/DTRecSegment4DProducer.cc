/** \class DTRecSegment4DProducer
 *  Builds the segments in the DT chambers.
 *
 *  $Date: $
 *  $Revision: $
 * \author Riccardo Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "RecoLocalMuon/DTSegment/src/DTRecSegment4DProducer.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "RecoLocalMuon/DTSegment/src/DTSegmentUpdator.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/DTDetIdAccessor.h"
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
  
  // Get the 2D rechits from the event
  Handle<DTRecSegment2DCollection> allHits; 
  event.getByLabel(theRecHits2DLabel, allHits);

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

  for (dtSuperLayerIt = allHits->id_begin(); dtSuperLayerIt != allHits->id_end(); ++dtSuperLayerIt){

    // The superLayerId
    DTSuperLayerId superLayerId = (*dtSuperLayerIt);

    // Check the DTChamberId
    const DTChamberId chId = superLayerId.chamberId();
    if (chId==oldChId) continue; // I'm on the same Chamber as before
    oldChId = chId;
    if(debug) cout << "ChamberId: "<< chId << endl;

    // Get the chamber
    const DTChamber *chamber = theGeom->chamber(chId);

    //Extract the DTRecSegment2DCollection ranges for the three different SL
    DTRecSegment2DCollection::range rangePhi1   = allHits->get(DTDetIdAccessor::bySuperLayer(DTSuperLayerId(chId,1)));
    DTRecSegment2DCollection::range rangeTheta = allHits->get(DTDetIdAccessor::bySuperLayer(DTSuperLayerId(chId,2)));
    DTRecSegment2DCollection::range rangePhi2   = allHits->get(DTDetIdAccessor::bySuperLayer(DTSuperLayerId(chId,3)));
    
    // Fill the DTRecSegment2D containers for the three different SL
    vector<DTRecSegment2D> segments2DPhi1(rangePhi1.first,rangePhi1.second);
    vector<DTRecSegment2D> segments2DTheta(rangeTheta.first,rangeTheta.second);
    vector<DTRecSegment2D> segments2DPhi2(rangePhi2.first,rangePhi2.second);

    if(debug)
      cout << "Number of 2D-segments in the first  SL (Phi)" << segments2DPhi1.size() << endl
	   << "Number of 2D-segments in the second SL (Theta)" << segments2DTheta.size() << endl
	   << "Number of 2D-segments in the third  SL (Phi)" << segments2DPhi2.size() << endl;
  

    cout << "Start 4D-Segments Reco " << endl;
    OwnVector<DTRecSegment4D> segments4D = the4DAlgo->reconstruct(chamber, 
								  segments2DPhi1,
								  segments2DTheta,
								  segments2DPhi2);
    
    cout << "Number of reconstructed 4D-segments" << segments4D.size() << endl;

    if (segments4D.size() > 0 )
      // convert the OwnVector into a Collection
      segments4DCollection->put(chId, segments4D.begin(),segments4D.end());
  }
  // Load the output in the Event
  event.put(segments4DCollection);
}

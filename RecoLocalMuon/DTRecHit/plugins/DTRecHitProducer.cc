/** \file
 *
 *  \author G. Cerminara
 */

#include "DTRecHitProducer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"

#include "RecoLocalMuon/DTRecHit/interface/DTRecHitBaseAlgo.h"
#include "RecoLocalMuon/DTRecHit/interface/DTRecHitAlgoFactory.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include <string>


using namespace edm;
using namespace std;




DTRecHitProducer::DTRecHitProducer(const ParameterSet& config) :
  // Set verbose output
  debug(config.getUntrackedParameter<bool>("debug", false))
{
  if(debug)
    cout << "[DTRecHitProducer] Constructor called" << endl;
  
  produces<DTRecHitCollection>();

  DTDigiToken_ = consumes<DTDigiCollection>(config.getParameter<InputTag>("dtDigiLabel"));

  // Get the concrete reconstruction algo from the factory
  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = DTRecHitAlgoFactory::get()->create(theAlgoName,
					       config.getParameter<ParameterSet>("recAlgoConfig"));
}

DTRecHitProducer::~DTRecHitProducer(){
  if(debug)
    cout << "[DTRecHitProducer] Destructor called" << endl;
  delete theAlgo;
}



void DTRecHitProducer::produce(Event& event, const EventSetup& setup) {
  // Get the DT Geometry
  ESHandle<DTGeometry> dtGeom;
  setup.get<MuonGeometryRecord>().get(dtGeom);

  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByToken(DTDigiToken_, digis);

  // Pass the EventSetup to the algo
  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits
  auto recHitCollection = std::make_unique<DTRecHitCollection>();


  // Iterate through all digi collections ordered by LayerId   
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin();
	   dtLayerIt != digis->end();
	   ++dtLayerIt){
    // The layerId
    const DTLayerId& layerId = (*dtLayerIt).first;
    // Get the GeomDet from the setup
    const DTLayer* layer = dtGeom->layer(layerId);

    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& range = (*dtLayerIt).second;
    
    OwnVector<DTRecHit1DPair> recHits =
      theAlgo->reconstruct(layer, layerId, range);
    
    if(debug)
      cout << "Number of hits in this layer: " << recHits.size() << endl;
    if(!recHits.empty()) //FIXME: is it really needed?
      recHitCollection->put(layerId, recHits.begin(), recHits.end());
  }

  event.put(std::move(recHitCollection));
}

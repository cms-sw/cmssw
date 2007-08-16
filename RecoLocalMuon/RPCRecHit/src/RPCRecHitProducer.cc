/** \file
 *
 *  $Date: 2006/07/13 12:04:31 $
 *  $Revision: 1.5 $
 *  \author M. Maggi -- INFN Bari
*/

#include "RPCRecHitProducer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"

#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitBaseAlgo.h"
#include "RecoLocalMuon/RPCRecHit/interface/RPCRecHitAlgoFactory.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <string>


using namespace edm;
using namespace std;




RPCRecHitProducer::RPCRecHitProducer(const ParameterSet& config){
  // Set verbose output

  produces<RPCRecHitCollection>();

  theRPCDigiLabel = config.getParameter<string>("rpcDigiLabel");
  
  // Get the concrete reconstruction algo from the factory
  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = RPCRecHitAlgoFactory::get()->create(theAlgoName,
						config.getParameter<ParameterSet>("recAlgoConfig"));
}

RPCRecHitProducer::~RPCRecHitProducer(){
  delete theAlgo;
}



void RPCRecHitProducer::produce(Event& event, const EventSetup& setup) {
  // Get the RPC Geometry
  ESHandle<RPCGeometry> rpcGeom;
  setup.get<MuonGeometryRecord>().get(rpcGeom);

  // Get the digis from the event
  Handle<RPCDigiCollection> digis; 
  event.getByLabel(theRPCDigiLabel,digis);

  // Pass the EventSetup to the algo
  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits
  auto_ptr<RPCRecHitCollection> recHitCollection(new RPCRecHitCollection());


  // Iterate through all digi collections ordered by LayerId   
  RPCDigiCollection::DigiRangeIterator rpcdgIt;
  for (rpcdgIt = digis->begin(); rpcdgIt != digis->end();
       ++rpcdgIt){
       
    // The layerId
    const RPCDetId& rpcId = (*rpcdgIt).first;
    // Get the GeomDet from the setup
    const RPCRoll* roll = rpcGeom->roll(rpcId);

    // Get the iterators over the digis associated with this LayerId
    const RPCDigiCollection::Range& range = (*rpcdgIt).second;
    
    OwnVector<RPCRecHit> recHits =
      theAlgo->reconstruct(*roll, rpcId, range);
    
    if(recHits.size() > 0) //FIXME: is it really needed?
      recHitCollection->put(rpcId, recHits.begin(), recHits.end());
  }

  event.put(recHitCollection);
}




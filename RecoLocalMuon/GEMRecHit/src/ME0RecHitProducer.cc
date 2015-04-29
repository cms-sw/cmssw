/** \file
 *
 *  $Date: 2013/04/24 17:16:35 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
*/

#include "ME0RecHitProducer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/GEMDigi/interface/ME0DigiPreRecoCollection.h"

#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"

#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitBaseAlgo.h"
#include "RecoLocalMuon/GEMRecHit/interface/ME0RecHitAlgoFactory.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHitCollection.h"

#include <string>


using namespace edm;
using namespace std;


ME0RecHitProducer::ME0RecHitProducer(const ParameterSet& config){

  produces<ME0RecHitCollection>();
  theME0DigiLabel = config.getParameter<InputTag>("me0DigiLabel");
  
  // Get the concrete reconstruction algo from the factory

  string theAlgoName = config.getParameter<string>("recAlgo");
  theAlgo = ME0RecHitAlgoFactory::get()->create(theAlgoName,
						config.getParameter<ParameterSet>("recAlgoConfig"));
}


ME0RecHitProducer::~ME0RecHitProducer(){
  delete theAlgo;
}



void ME0RecHitProducer::beginRun(const edm::Run& r, const edm::EventSetup& setup){
}



void ME0RecHitProducer::produce(Event& event, const EventSetup& setup) {

  // Get the ME0 Geometry
  ESHandle<ME0Geometry> me0Geom;
  setup.get<MuonGeometryRecord>().get(me0Geom);

  // Get the digis from the event

  Handle<ME0DigiPreRecoCollection> digis; 
  event.getByLabel(theME0DigiLabel,digis);

  // Pass the EventSetup to the algo

  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits

  auto_ptr<ME0RecHitCollection> recHitCollection(new ME0RecHitCollection());

  // Iterate through all digi collections ordered by LayerId   

  ME0DigiPreRecoCollection::DigiRangeIterator me0dgIt;
  for (me0dgIt = digis->begin(); me0dgIt != digis->end();
       ++me0dgIt){
       
    // The layerId
    const ME0DetId& me0Id = (*me0dgIt).first;

    // Get the GeomDet from the setup
    //    const ME0EtaPartition* roll = me0Geom->etaPartition(me0Id);

    // Get the iterators over the digis associated with this LayerId
    const ME0DigiPreRecoCollection::Range& range = (*me0dgIt).second;

    // Call the reconstruction algorithm    

    OwnVector<ME0RecHit> recHits =
      theAlgo->reconstruct(me0Id, range);
    
    if(recHits.size() > 0) //FIXME: is it really needed?
      recHitCollection->put(me0Id, recHits.begin(), recHits.end());
  }

  event.put(recHitCollection);

}


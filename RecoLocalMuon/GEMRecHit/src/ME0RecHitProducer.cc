/** \file
 *
 *  $Date: 2013/04/24 17:16:35 $
 *  $Revision: 1.1 $
 *  \author M. Maggi -- INFN Bari
*/

#include "ME0RecHitProducer.h"


ME0RecHitProducer::ME0RecHitProducer(const edm::ParameterSet& config){

  produces<ME0RecHitCollection>();
 
  m_token = consumes<ME0DigiPreRecoCollection>( config.getParameter<edm::InputTag>("me0DigiLabel") ); 

 
  // Get the concrete reconstruction algo from the factory

  std::string theAlgoName = config.getParameter<std::string>("recAlgo");
  theAlgo = ME0RecHitAlgoFactory::get()->create(theAlgoName,
						config.getParameter<edm::ParameterSet>("recAlgoConfig"));
}


ME0RecHitProducer::~ME0RecHitProducer(){
  delete theAlgo;
}



void ME0RecHitProducer::beginRun(const edm::Run& r, const edm::EventSetup& setup){
}



void ME0RecHitProducer::produce(edm::Event& event, const edm::EventSetup& setup) {

  // Get the ME0 Geometry
  edm::ESHandle<ME0Geometry> me0Geom;
  setup.get<MuonGeometryRecord>().get(me0Geom);

  // Get the digis from the event

  edm::Handle<ME0DigiPreRecoCollection> digis; 
  event.getByToken(m_token,digis);

  // Pass the EventSetup to the algo

  theAlgo->setES(setup);

  // Create the pointer to the collection which will store the rechits

  auto recHitCollection = std::make_unique<ME0RecHitCollection>();

  // Iterate through all digi collections ordered by LayerId   

  ME0DigiPreRecoCollection::DigiRangeIterator me0dgIt;
  for (me0dgIt = digis->begin(); me0dgIt != digis->end();
       ++me0dgIt){
       
    // The layerId
    const ME0DetId& me0Id = (*me0dgIt).first;


    // Get the iterators over the digis associated with this LayerId
    const ME0DigiPreRecoCollection::Range& range = (*me0dgIt).second;

    // Call the reconstruction algorithm    

    edm::OwnVector<ME0RecHit> recHits =
      theAlgo->reconstruct(me0Id, range);
    
    if(!recHits.empty())
      recHitCollection->put(me0Id, recHits.begin(), recHits.end());
  }

  event.put(std::move(recHitCollection));

}


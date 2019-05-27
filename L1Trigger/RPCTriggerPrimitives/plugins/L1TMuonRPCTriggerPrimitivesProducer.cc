// RPCTriggerPrimitives producer for RPPRecHits in L1T Level
// Author Alejandro Segura -- Universidad de los Andes

#include "L1TMuonRPCTriggerPrimitivesProducer.h"

L1TMuonRPCTriggerPrimitivesProducer::L1TMuonRPCTriggerPrimitivesProducer(const edm::ParameterSet& iConfig): 
  preprocess_pointer_(std::make_unique<PrimitivePreprocess>(iConfig, consumesCollector())){

  produces<RPCRecHitCollection>();

}

L1TMuonRPCTriggerPrimitivesProducer::~L1TMuonRPCTriggerPrimitivesProducer(){
}

void L1TMuonRPCTriggerPrimitivesProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  // Create pointers to the collections which will store the new primitive digis
  auto Tprimitive_digis = std::make_unique<RPCRecHitCollection>();
 

  preprocess_pointer_->beginRun(iSetup);  
  preprocess_pointer_->Preprocess(iEvent, iSetup, *Tprimitive_digis);
  // Fill the output collections
  iEvent.put(std::move(Tprimitive_digis));
  
}

void L1TMuonRPCTriggerPrimitivesProducer::beginStream(edm::StreamID iID){
}

void L1TMuonRPCTriggerPrimitivesProducer::endStream(){
}

// Define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonRPCTriggerPrimitivesProducer);


// Emulator that takes RPC hits and produces CPPFDigis to send to EMTF
// Author Alejandro Segura -- Universidad de los Andes

#include "L1TMuonCPPFDigiProducer.h"


L1TMuonCPPFDigiProducer::L1TMuonCPPFDigiProducer(const edm::ParameterSet& iConfig) :
  cppf_emulator_(std::make_unique<EmulateCPPF>(iConfig, consumesCollector()))
  //cppf_emulator_(new EmulateCPPF(iConfig, consumesCollector()))
{
  // produces<l1t::CPPFDigiCollection>("rpcDigi");
  produces<l1t::CPPFDigiCollection>("recHit");
}

L1TMuonCPPFDigiProducer::~L1TMuonCPPFDigiProducer() {
}

void L1TMuonCPPFDigiProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
    
  // Create pointers to the collections which will store the cppfDigis 
  // auto cppf_rpcDigi = std::make_unique<l1t::CPPFDigiCollection>();
  auto cppf_recHit  = std::make_unique<l1t::CPPFDigiCollection>();

  // Main CPPF emulation process: emulates CPPF output from RPCDigi or RecHit inputs
  // From src/EmulateCPPF.cc
  // cppf_emulator_->process(iEvent, iSetup, *cppf_rpcDigi, *cppf_recHit);
  cppf_emulator_->process(iEvent, iSetup, *cppf_recHit);

  // Fill the output collections
  // iEvent.put(std::move(cppf_rpcDigi), "rpcDigi");
  iEvent.put(std::move(cppf_recHit),  "recHit");
}

void L1TMuonCPPFDigiProducer::beginStream(edm::StreamID iID) {
}

void L1TMuonCPPFDigiProducer::endStream(){
}

// Define this as a plug-in
DEFINE_FWK_MODULE(L1TMuonCPPFDigiProducer);

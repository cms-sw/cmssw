
#include "L1Trigger/L1TMuonCPPF/interface/EmulateCPPF.h"
#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include <string>
#include <fstream>

EmulateCPPF::EmulateCPPF(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes) :
  // rpcDigi_processors_(),
  recHit_processors_(),
  // rpcDigiToken_( iConsumes.consumes<RPCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("recHitLabel")) ),
  recHitToken_(iConsumes.consumes<RPCRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitLabel"))),
  cppfSource_(CppfSource::EventSetup)
{
  
  const std::string cppfSource = iConfig.getParameter<std::string>("cppfSource");
  //Look up table
  if (cppfSource == "File"){
    std::cout << "Running with the Look up table" << std::endl; 
    cppfSource_ = CppfSource::File;
    edm::FileInPath fp = iConfig.getParameter<edm::FileInPath>("cppfvecfile");
    std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
    if ( !inputFile ) {
      std::cerr << "CPPF look up table file cannot not be opened" << std::endl;
      exit(1);
    }
    while ( inputFile.good() ) {
      RecHitProcessor::CppfItem Item;
      inputFile >> Item.rawId >> Item.strip >> Item.lb >> Item.halfchannel >> Item.int_phi >> Item.int_theta;
      if ( inputFile.good() ) CppfVec_1.push_back(Item);
    }
    inputFile.close();
  }
  
  //RPC Geometry	
  else if (cppfSource == "Geo") {
    std::cout << "Running with the RPC Geometry" << std::endl;
    cppfSource_ = CppfSource::EventSetup;
  }
  //Error for wrong input
  else {
    std::cerr << "Error: Specify in python/simCppfDigis_cfi.py 'File' for look up  table or 'Geo' for RPC Geometry" << std::endl;
    exit(1); 
  }
  
}

EmulateCPPF::~EmulateCPPF() {
}

void EmulateCPPF::process(
			  const edm::Event& iEvent, const edm::EventSetup& iSetup,
			  // l1t::CPPFDigiCollection& cppf_rpcDigi,
			  l1t::CPPFDigiCollection& cppf_recHit
			  ) {
  
  if( cppfSource_ == CppfSource::File ){
    //Using the look up table to fill the information
    cppf_recHit.clear();
    for (unsigned int iBoard = 0; iBoard < recHit_processors_.size(); iBoard++) {
      recHit_processors_.at(iBoard).processLook( iEvent, iSetup, recHitToken_, CppfVec_1, cppf_recHit );
    }
    return; 
  }
  if (cppfSource_ == CppfSource::EventSetup) {
    // Clear output collections
    // cppf_rpcDigi.clear();
    cppf_recHit.clear();
    
    // // Get the RPCDigis from the event
    // edm::Handle<RPCTag::digi_collection> rpcDigis;
    // iEvent.getByToken(rpcDigiToken_, rpcDigis);
    
    // _________________________________________________________________________________
    // Run the CPPF clusterization+coordinate conversion algo on RPCDigis and RecHits
    
    // For now, treat CPPF as single board
    // In the future, may want to treat the 4 CPPF boards in each endcap as separate entities
    
    // for (unsigned int iBoard = 0; iBoard < rpcDigi_processors_.size(); iBoard++) {
    // rpcDigi_processors_.at(iBoard).process( iSetup, rpcDigis, cppf_rpcDigi );
    // }
    for (unsigned int iBoard = 0; iBoard < recHit_processors_.size(); iBoard++) {
      recHit_processors_.at(iBoard).process( iEvent, iSetup, recHitToken_, cppf_recHit );
    } 
    return;
  }
} // End void EmulateCPPF::process()					   

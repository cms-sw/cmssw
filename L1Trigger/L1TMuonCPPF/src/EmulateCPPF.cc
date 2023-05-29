
#include "CondFormats/RPCObjects/interface/RPCDeadStrips.h"
#include "CondFormats/RPCObjects/interface/RPCMaskedStrips.h"
#include "L1Trigger/L1TMuonCPPF/interface/EmulateCPPF.h"
#include <fstream>
#include <string>
#include "L1Trigger/L1TMuonCPPF/src/CPPFClusterContainer.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFCluster.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFClusterizer.h"
#include "L1Trigger/L1TMuonCPPF/src/CPPFMaskReClusterizer.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "DataFormats/Common/interface/DetSetVector.h"

EmulateCPPF::EmulateCPPF(const edm::ParameterSet &iConfig, edm::ConsumesCollector &&iConsumes)
    : recHit_processors_(),
      rpcDigiToken_(iConsumes.consumes<RPCDigiCollection>(iConfig.getParameter<edm::InputTag>("rpcDigiLabel"))),
      recHitToken_(iConsumes.consumes<RPCRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitLabel"))),
      rpcDigiSimLinkToken_(iConsumes.consumes<edm::DetSetVector<RPCDigiSimLink> >(
          iConfig.getParameter<edm::InputTag>("rpcDigiSimLinkLabel"))),
      rpcGeomToken_(iConsumes.esConsumes()),
      cppfSource_(CppfSource::EventSetup),
      MaxClusterSize_(0) {
  MaxClusterSize_ = iConfig.getParameter<int>("MaxClusterSize");
  const std::string cppfSource = iConfig.getParameter<std::string>("cppfSource");
  //      Look up table
  if (cppfSource == "File") {
    cppfSource_ = CppfSource::File;
    edm::FileInPath fp = iConfig.getParameter<edm::FileInPath>("cppfvecfile");
    std::ifstream inputFile(fp.fullPath().c_str(), std::ios::in);
    if (!inputFile) {
      throw cms::Exception("No LUT") << "Error: CPPF look up table file cannot not be opened";
      exit(1);
    }
    while (inputFile.good()) {
      RecHitProcessor::CppfItem Item;
      inputFile >> Item.rawId >> Item.strip >> Item.lb >> Item.halfchannel >> Item.int_phi >> Item.int_theta;
      if (inputFile.good())
        CppfVec_1.push_back(Item);
    }
    inputFile.close();
  }
  //      RPC Geometry
  else if (cppfSource == "Geo") {
    cppfSource_ = CppfSource::EventSetup;
  }
  //      Error for wrong input
  else {
    throw cms::Exception("Invalid option")
        << "Error: Specify in python/emulatorCppfDigis_cfi 'File' for look up table or 'Geo' for RPC Geometry";
    exit(1);
  }
}

EmulateCPPF::~EmulateCPPF() {}

void EmulateCPPF::process(const edm::Event &iEvent,
                          const edm::EventSetup &iSetup,
                          l1t::CPPFDigiCollection &cppf_recHit) {
  if (cppfSource_ == CppfSource::File) {  // Using the look up table to fill the information
    cppf_recHit.clear();
    for (auto &recHit_processor : recHit_processors_) {
      recHit_processor.processLook(iEvent,
                                   iSetup,
                                   recHitToken_,
                                   rpcDigiToken_,
                                   rpcDigiSimLinkToken_,
                                   rpcGeomToken_,
                                   CppfVec_1,
                                   cppf_recHit,
                                   MaxClusterSize_);
    }
  } else if (cppfSource_ == CppfSource::EventSetup) {
    // Clear output collections
    // cppf_rpcDigi.clear();
    cppf_recHit.clear();

    //Get the RPCDigis from the event
    // Run the CPPF clusterization+coordinate conversion algo on RPCDigis and
    // RecHits

    // For now, treat CPPF as single board
    // In the future, may want to treat the 4 CPPF boards in each endcap as
    // separate entities

    // for (unsigned int iBoard = 0; iBoard < rpcDigi_processors_.size();
    // iBoard++) { rpcDigi_processors_.at(iBoard).process( iSetup, rpcDigis,
    // cppf_rpcDigi );
    // }
    for (auto &recHit_processor : recHit_processors_) {
      recHit_processor.process(
          iEvent, iSetup, recHitToken_, rpcDigiToken_, rpcDigiSimLinkToken_, rpcGeomToken_, cppf_recHit);
    }
  }
}  // End void EmulateCPPF::process()

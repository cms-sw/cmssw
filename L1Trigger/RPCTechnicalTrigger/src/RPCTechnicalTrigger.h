// -*- C++ -*-
//
// Package:    RPCTechnicalTrigger
// Class:      RPCTechnicalTrigger
// 
/**\class RPCTechnicalTrigger RPCTechnicalTrigger.cc L1Trigger/RPCTechnicalTrigger/src/RPCTechnicalTrigger.cc

Description: Implementation of the RPC Technical Trigger Emulator

Implementation:


*/
//
// Original Author:  Andres Osorio
//         Created:  Tue Mar 10 13:59:40 CET 2009
//
//
//
//
// $Id: 
//
//

#ifndef RPCTECHNICALTRIGGER_H 
#define RPCTECHNICALTRIGGER_H 1

// system include files
#include <memory>
#include <bitset>

// Include files From CMSSW

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"

// Local to project
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/TTUEmulator.h"

#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/DataRecord/interface/RBCBoardSpecsRcd.h"

#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"
#include "CondFormats/DataRecord/interface/TTUBoardSpecsRcd.h"

//...........................................................................

class RPCTechnicalTrigger : public edm::EDProducer {
public:

  explicit RPCTechnicalTrigger(const edm::ParameterSet&);
  ~RPCTechnicalTrigger();
  
private:
  
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  //...........................................................................
  
  void printinfo();
  
  TTUEmulator * m_ttu[3];
  
  RPCInputSignal * m_input;
  
  ProcessInputSignal * m_signal;
  
  std::bitset<1> m_trigger;
  std::bitset<6> m_triggerbits;
  std::bitset<8> m_gmtfilter;
  std::bitset<5> m_bits_24o28;
  
  edm::ESHandle<RPCGeometry> m_rpcGeometry;
  
  int m_debugmode;
  int m_triggermode;
  
  std::string m_testfile;
  std::vector<unsigned>    m_ttBits;
  std::vector<std::string> m_ttNames;
  
  edm::InputTag m_GMTLabel;
  
  const TTUBoardSpecs * m_ttuspecs;
  const RBCBoardSpecs * m_rbcspecs;
  
  int m_ievt;
  int m_cand;
  
};

#endif // RPCTECHNICALTRIGGER_H

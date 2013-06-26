// -*- C++ -*-
//
// Package:    RPCTechnicalTrigger
// Class:      RPCTechnicalTrigger
// 
/**\class RPCTechnicalTrigger RPCTechnicalTrigger.cc L1Trigger/RPCTechnicalTrigger/interface/RPCTechnicalTrigger.cc

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

// Local to project
#include "L1Trigger/RPCTechnicalTrigger/interface/ProcessInputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUConfigurator.h"

#include "CondFormats/RPCObjects/interface/RBCBoardSpecs.h"
#include "CondFormats/DataRecord/interface/RBCBoardSpecsRcd.h"
#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"
#include "CondFormats/DataRecord/interface/TTUBoardSpecsRcd.h"

//Technical trigger bits
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTrigger.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"


//...........................................................................

class RPCTechnicalTrigger : public edm::EDProducer {
public:

  explicit RPCTechnicalTrigger(const edm::ParameterSet&);
  ~RPCTechnicalTrigger();
  
private:
  
  //virtual void beginJob() ;
  virtual void beginRun(edm::Run const&, const edm::EventSetup&) override final;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  //...........................................................................
  
  void printinfo();

  bool Reset();
    
  TTUEmulator * m_ttu[3];

  TTUEmulator * m_ttuRbcLine[3];
  
  RPCInputSignal * m_input;
  
  ProcessInputSignal * m_signal;
  
  std::bitset<5> m_triggerbits;
  
  edm::ESHandle<RPCGeometry> m_rpcGeometry;
  
  int m_verbosity;
  int m_useEventSetup;
  std::string m_configFile;
  std::vector<unsigned> m_ttBits;
  std::vector<std::string> m_ttNames;
  edm::InputTag m_rpcDigiLabel;
  edm::InputTag m_rpcSimLinkInstance;
  
  int m_useRPCSimLink;
  
  TTUConfigurator     * m_readConfig;
  const TTUBoardSpecs * m_ttuspecs;
  const RBCBoardSpecs * m_rbcspecs;
  
  int m_ievt;
  int m_cand;
  int m_boardIndex[3];
  int m_nWheels[3];
  int m_maxTtuBoards;
  int m_maxBits;
  bool m_hasConfig;
  
  class TTUResults {
  public:
    TTUResults() {;}
    
    TTUResults( int idx, int bx, int wh1, int wh2 ):
      m_ttuidx(idx),
      m_bx(bx),
      m_trigWheel1(wh1),
      m_trigWheel2(wh2) {;}
    
    TTUResults( int idx, int bx, int wh1, int wh2 , int wdg ):
      m_ttuidx(idx),
      m_bx(bx),
      m_trigWheel1(wh1),
      m_trigWheel2(wh2),
      m_wedge(wdg) {;}
    
    ~TTUResults() 
    {
      m_ttuidx=0;
      m_bx=0;
      m_trigWheel1=0;
      m_trigWheel2=0;
      m_wedge=0;
    }
    
    int m_ttuidx;
    int m_bx;
    int m_trigWheel1;
    int m_trigWheel2;
    int m_wedge;
  
    int getTriggerForWheel( int wheel ) 
    {
      if( abs(wheel) > 1 ) return m_trigWheel2;
      else return m_trigWheel1;
    }
    
  
  };
  
  struct sortByBx {
    bool operator()( const TTUResults * a, const TTUResults * b )
    {
      return (*a).m_bx < (*b).m_bx;
    }
  };
  
  std::vector<TTUResults*> m_serializedInfoLine1;
  std::vector<TTUResults*> m_serializedInfoLine2;

  int convertToMap( const std::vector<TTUResults*> & );
  
  bool searchCoincidence( int , int );
  
  std::map<int,int> m_WheelTtu;
  
  std::map<int,TTUResults*> m_ttuResultsByQuadrant;

  std::vector<int> m_quadrants;
  
  std::vector<int>::iterator m_firstSector;
  
    
};

#endif // RPCTECHNICALTRIGGER_H

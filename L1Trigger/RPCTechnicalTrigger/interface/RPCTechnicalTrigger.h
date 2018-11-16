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
#include <array>

// Include files From CMSSW

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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

class RPCTechnicalTrigger : public edm::stream::EDProducer<> {
public:

  explicit RPCTechnicalTrigger(const edm::ParameterSet&);
  ~RPCTechnicalTrigger() override;
  
private:
  
  //virtual void beginJob() ;
  void beginRun(edm::Run const&, const edm::EventSetup&) final;
  void produce(edm::Event&, const edm::EventSetup&) override;

  //...........................................................................
  
  void printinfo() const;

  static constexpr int kMaxTtuBoards = 3;
  std::array<TTUEmulator,kMaxTtuBoards> m_ttu;

  std::array<TTUEmulator,kMaxTtuBoards> m_ttuRbcLine;
    
  const int m_verbosity;
  const int m_useEventSetup;
  std::string m_configFile;
  const std::vector<unsigned> m_ttBits;
  const std::vector<std::string> m_ttNames;
  const edm::InputTag m_rpcDigiLabel;
  const edm::EDGetTokenT<RPCDigiCollection> m_rpcDigiToken;
  
  const int m_useRPCSimLink;
  
  std::unique_ptr<TTUConfigurator> m_readConfig;
  const TTUBoardSpecs * m_ttuspecs;
  const RBCBoardSpecs * m_rbcspecs;
  
  bool m_hasConfig;
  
  class TTUResults {
  public:
    TTUResults() = default;
    TTUResults(const TTUResults&) = default;
    TTUResults(TTUResults&&) = default;
    TTUResults& operator=(TTUResults const&) = default;
    TTUResults& operator=(TTUResults&&) = default;
    
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
    
    int m_ttuidx;
    int m_bx;
    int m_trigWheel1;
    int m_trigWheel2;
    int m_wedge;
  
    int getTriggerForWheel( int wheel ) const
    {
      if( abs(wheel) > 1 ) return m_trigWheel2;
      else return m_trigWheel1;
    }
    
  
  };
  

  std::map<int,TTUResults*> convertToMap( const std::vector<std::unique_ptr<TTUResults>> & ) const;
  
  bool searchCoincidence( int , int, std::map<int, TTUResults*> const& ttuResultsByQuandrant ) const;
};

#endif // RPCTECHNICALTRIGGER_H

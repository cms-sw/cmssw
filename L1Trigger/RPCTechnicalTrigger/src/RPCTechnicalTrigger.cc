// -*- C++ -*-
//
// Package:    RPCTechnicalTrigger
// Class:      RPCTechnicalTrigger
// 
/**\class RPCTechnicalTrigger RPCTechnicalTrigger.cc L1Trigger/RPCTechnicalTrigger/src/RPCTechnicalTrigger.cc
   
Description: 

RPC Technical Trigger Emulator: the RPCTechnicalTrigger is the main controler

Implementation:

*/
//
// Original Author:  Andres Felipe Osorio Oliveros
//         Created:  Thu Nov 13 08:21:16 CET 2008
// $Id: RPCTechnicalTrigger.cc,v 1.1 2009/01/30 15:42:48 aosorio Exp $

// local
#include "L1Trigger/RPCTechnicalTrigger/src/RPCTechnicalTrigger.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RPCProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RPCProcessDigiSignal.h"

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig) : 
  edm::EDAnalyzer()
{
  
  m_debugmode    = iConfig.getUntrackedParameter<int>("RPCTTDebugMode",0);
  
  m_rbclogictype = iConfig.getUntrackedParameter<std::string>("RBCLogicType",
                                                              std::string("TestLogic"));
  
  m_ttulogictype = iConfig.getUntrackedParameter<std::string>("TTULogicType",
                                                              std::string("TrackingAlg"));
  
  //... Debug mode:
  // 0 - no debug
  // 1 - human readable debug
  // 2 - technical debug  
  
  if ( m_debugmode != 1 && m_debugmode != 2) m_debugmode = 0;
  
  //...............................................................
  
  if ( m_debugmode == 1 ) {
    //. Read RBC input for an ascii file
    
    std::string tmpfile = iConfig.getUntrackedParameter<std::string>("TmpDataFile",
                                                                     std::string("testdata.txt"));
    
    m_signal  = dynamic_cast<ProcessInputSignal*>(new RPCProcessTestSignal( tmpfile.c_str()));
    
  }
  
  //if ( m_debugmode != 0 ) {
  //  
  //  m_ttu[0] = new TTUEmulator( 1 , m_rbclogictype.c_str(), m_ttulogictype.c_str(), 2 );
  //  m_ttu[1] = new TTUEmulator( 2 , m_rbclogictype.c_str(), m_ttulogictype.c_str(), 2 );
  //  m_ttu[2] = new TTUEmulator( 3 , m_rbclogictype.c_str(), m_ttulogictype.c_str(), 1 );
  //  
  //  return;
  //} 
  
  //... There are three Technical Trigger Units: 1 per 2 Wheels

  m_ttu[0] = new TTUEmulator( 1 , 2 );
  m_ttu[1] = new TTUEmulator( 3 , 2 );
  m_ttu[2] = new TTUEmulator( 2 , 1 );
  
  //...............................................................
  
}

//=============================================================================
// Destructor
//=============================================================================
RPCTechnicalTrigger::~RPCTechnicalTrigger() {
  
  delete m_ttu[0];
  delete m_ttu[1];
  delete m_ttu[2];

  if ( m_signal ) delete m_signal;
  
} 

//=============================================================================
void RPCTechnicalTrigger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  edm::Handle<RPCDigiCollection> pIn;
  iEvent.getByLabel("muonRPCDigis",pIn);
  
  if ( m_debugmode != 1){
    m_signal  = dynamic_cast<ProcessInputSignal*>(new RPCProcessDigiSignal( m_rpcGeometry, pIn ));
  }
  
  bool status(false);
  status = m_signal->next();
  if ( !status) return;
  
  RPCInputSignal * input = m_signal->retrievedata();
  
  //. distribute data to different TTU emulators and process it
  
  for(int k=0; k<3; ++k) {
    
    if ( m_debugmode == 1 ) {
      m_ttu[k]->processlocal( input );
      continue;
    } 
    
    m_ttu[k]->processglobal( input );
    
  }
  
  //.. analyse results from emulation

  
  
  //... reset data map for next event
  
  input->clear();
  
  //...
  
  if( m_debugmode != 1 ){
    delete m_signal;
    m_signal = NULL;
  }
  
}

// ------------ method called once each job just before starting event loop  ------------
void RPCTechnicalTrigger::beginJob(const edm::EventSetup& _evtSetup)
{

  bool status(false);

  std::cout << "RPCTechnicalTrigger::beginJob>" << std::endl;

  //.   Set up RPC geometry
  
  _evtSetup.get<MuonGeometryRecord>().get( m_rpcGeometry );
  
  //..  Get Board Specifications (hardware configuration)
  
  edm::ESHandle<RBCBoardSpecs> pRBCSpecs;
  _evtSetup.get<RBCBoardSpecsRcd>().get(pRBCSpecs);
  m_rbcspecs = pRBCSpecs.product();
  
  edm::ESHandle<TTUBoardSpecs> pTTUSpecs;
  _evtSetup.get<TTUBoardSpecsRcd>().get(pTTUSpecs);
  
  m_ttuspecs = pTTUSpecs.product();
    
  for (int k=0; k < 3; ++k )
    m_ttu[k]->setSpecifications( m_ttuspecs, m_rbcspecs );
  
  //... Initialize all
  
  for (int k=0; k < 3; ++k )
    status = m_ttu[k]->initialise();
  
  //..........
  
  std::cout << "RPCTechnicalTrigger::beginJob> pointers to specs " 
            <<  &pRBCSpecs << " " << m_rbcspecs << std::endl;
  
  std::cout << "RPCTechnicalTrigger::beginJob> pointers to specs " 
            <<  &pTTUSpecs << " " << m_ttuspecs << std::endl;
  
  //..........
  
}

void RPCTechnicalTrigger::printinfo()
{
  
  for (int k=0; k < 3; ++k )
    m_ttu[k]->printinfo();
  
}

// ------------ method called once each job just after ending the event loop  ------------
void RPCTechnicalTrigger::endJob() 
{
  std::cout << "RPCTechnicalTrigger::endJob>" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCTechnicalTrigger);

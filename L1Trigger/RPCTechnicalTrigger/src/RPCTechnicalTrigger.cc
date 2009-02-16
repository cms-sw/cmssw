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
// $Id: RPCTechnicalTrigger.cc,v 1.2 2009/02/05 13:46:21 aosorio Exp $

// local
#include "L1Trigger/RPCTechnicalTrigger/src/RPCTechnicalTrigger.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RPCProcessTestSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RPCProcessDigiSignal.h"

//Data Formats
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
RPCTechnicalTrigger::RPCTechnicalTrigger(const edm::ParameterSet& iConfig) : 
  edm::EDAnalyzer()
{

  edm::Service<TFileService> fs;
    
  m_debugmode    = iConfig.getUntrackedParameter<int>("RPCTTDebugMode",0);
  
  m_validation   = iConfig.getUntrackedParameter<int>("RPCTTValidationMode", 0);
  
  m_rbclogictype = iConfig.getUntrackedParameter<std::string>("RBCLogicType",
                                                              std::string("TestLogic"));
  
  m_ttulogictype = iConfig.getUntrackedParameter<std::string>("TTULogicType",
                                                              std::string("TrackingAlg"));
  
  m_GMTLabel     = iConfig.getUntrackedParameter<edm::InputTag>("GMTInputTag", edm::InputTag("gmt"));
  
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
    
    m_signal  = dynamic_cast<ProcessInputSignal*>(new RPCProcessTestSignal( tmpfile.c_str()) );
    
  }
  
  //... There are three Technical Trigger Units: 1 per 2 Wheels

  m_ttu[0] = new TTUEmulator( 1 , 2 );
  m_ttu[1] = new TTUEmulator( 2 , 1 );
  m_ttu[2] = new TTUEmulator( 3 , 2 );
  
  //... histograms

  m_houtput = new HistoOutput( fs , "Plots");
  
  //...........................................................................

  m_ievt = 0;
  m_cand = 0;
  
}

//=============================================================================
// Destructor
//=============================================================================
RPCTechnicalTrigger::~RPCTechnicalTrigger() {


  std::cout << "RPCTechnicalTrigger: object starts deletion" << std::endl;
  
  delete m_ttu[0];
  delete m_ttu[1];
  delete m_ttu[2];
  
  if ( m_debugmode == 1) {
    if (m_signal) delete m_signal;
  }
  
  if ( m_houtput ) delete m_houtput;
  
  std::cout << "RPCTechnicalTrigger: object deleted" << std::endl;
    
} 

//=============================================================================
void RPCTechnicalTrigger::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  edm::Handle<RPCDigiCollection> pIn;
  try {
    iEvent.getByLabel("muonRPCDigis", pIn);
  }
  catch (...) {
    edm::LogInfo("RPCTechnicalTrigger::analyze") << "can't find L1MuGMTReadoutCollection with label" << '\n';
    return;
  }
  
  if ( m_debugmode != 1){
    edm::LogInfo("RPCTechnicalTrigger::analyze") << "setting up a new signal object" << '\n';
    m_signal  = dynamic_cast<ProcessInputSignal*>(new RPCProcessDigiSignal( m_rpcGeometry, pIn ));
    std::cout << "RPCTechnicalTrigger::" << std::endl;
  }
  
  bool status(false);
  status = m_signal->next();
  if ( !status) return;
  
  RPCInputSignal * input = m_signal->retrievedata();
  
  //. distribute data to different TTU emulators and process it
  
  m_triggerbits.reset();
  
  int indx=0;
  
  for(int k=0; k<3; ++k) {
    
    indx=k*2;
    
    if ( m_debugmode == 1 ) {
      m_ttu[k]->processlocal( input );
      m_triggerbits.set( indx   , m_ttu[k]->m_trigger[0] );
      m_triggerbits.set( indx+1 , m_ttu[k]->m_trigger[1] );
      continue;
    } 
    
    m_ttu[k]->processglobal( input );
    m_triggerbits.set( indx   , m_ttu[k]->m_trigger[0] );
    m_triggerbits.set( indx+1 , m_ttu[k]->m_trigger[1] );
    
  }
  
  //.. analyse results from emulation
  
  int flag=0;
  m_trigger.reset();
  
  for(int k=0; k<6; ++k) {
    bool tmp = m_triggerbits[k];
    m_trigger[0] = ( m_trigger[0] || tmp );
  }
  
  if ( m_validation == 1 )
    validate ( iEvent, iSetup, flag );
    



  //... reset data map for next event
  
  input->clear();
  
  //...
  
  if( m_debugmode != 1 ){
    delete m_signal;
  }
  
  ++m_ievt;
  
  std::cout << "RPCTechnicalTrigger: end of analyze" << std::endl;
  
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

// ------------ method called once each job just after ending the event loop  ------------
void RPCTechnicalTrigger::endJob() 
{
  
  std::cout << "RPCTechnicalTrigger::endJob>" << std::endl;
  
}

void RPCTechnicalTrigger::printinfo()
{
  
  for (int k=0; k < 3; ++k )
    m_ttu[k]->printinfo();
  
}

// .......................................................................................

void RPCTechnicalTrigger::validate( const edm::Event& iEvent, const edm::EventSetup& iSetup, int & outflag )
{
  
  //.
  int nmucand = discriminateGMT( iEvent, iSetup );
  m_houtput->hnumGMTmuons->Fill( nmucand );
  
  m_valflag = 0;
  
  //..
  if ( nmucand > 0 ) {
    
    if ( m_gmtfilter[3] && m_trigger[0] ) m_valflag = 2;
    else if ( m_gmtfilter[3] && ! m_trigger[0] ) m_valflag = 4;
    else if ( ! m_gmtfilter[3] && m_trigger[0] ) m_valflag = 6;
    else m_valflag = 8;
    
  }
  
  m_houtput->hcompflag->Fill( m_valflag );
  
  //...
  
  
}

int RPCTechnicalTrigger::discriminateGMT( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  
  //...this a GMT trigger filter based on Anna Cimmino's DQMOffline/src/RPCTriggerFilter code
  edm::Handle<L1MuGMTReadoutCollection> pCollection;
  try {
    iEvent.getByLabel(m_GMTLabel.label(),pCollection); }
  catch (...) {
    edm::LogInfo("discriminateGMT") << "can't find L1MuGMTReadoutCollection with label "
                                    << m_GMTLabel.label() ;
    return -1; }
  
  bool rpcBarFlag  = false;
  bool rpcFwdFlag  = false;
  bool dtFlag      = false;
  bool cscFlag     = false;
  
  // get GMT readout collection
  const L1MuGMTReadoutCollection * gmtrc = pCollection.product();
  // get record vector
  std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
  // loop over records of individual bx's
  std::vector<L1MuGMTReadoutRecord>::const_iterator RRItr;
  
  int nmuons=0;
  
  edm::LogInfo("DiscriminateGMT") << "nRecords: " << gmt_records.size() << '\n';
  
  for( RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++ ) {
    
    std::vector<L1MuGMTExtendedCand> GMTCands   = RRItr->getGMTCands();
    
    int BxInEvent = RRItr->getBxInEvent();
    if(BxInEvent!=0) continue;
    
    //loop over GMT candidates in each record 
    std::vector<L1MuGMTExtendedCand>::const_iterator GMTItr;
    
    for( GMTItr = GMTCands.begin(); GMTItr != GMTCands.end(); ++GMTItr ) {
      ++nmuons;
      ++m_cand;
      
      if(GMTItr->empty()) continue;
      
      if (GMTItr->isRPC()&& !GMTItr->isFwd()) rpcBarFlag= true;
      else if (GMTItr->isRPC()) rpcFwdFlag=true;
      else if (!GMTItr->isFwd())dtFlag=true;
      else cscFlag=true;
      
    }
  }
  
  edm::LogInfo ("DiscriminateGMT") << "Number of GMT muons: " << nmuons;
  
  bool test(false);
  m_gmtfilter.reset();
  
  test = (rpcBarFlag || rpcFwdFlag) && !dtFlag && !cscFlag;  // rpc Event
  m_gmtfilter.set(0,test);
  test = !rpcFwdFlag && rpcBarFlag  && !dtFlag && !cscFlag;  // rpc Event in barrel
  m_gmtfilter.set(1,test);
  test = !rpcBarFlag && rpcFwdFlag  && !dtFlag && !cscFlag;  // rpc event in endcaps
  m_gmtfilter.set(2,test);
  test = !rpcFwdFlag && !rpcBarFlag && dtFlag  && !cscFlag;  // dt  event
  m_gmtfilter.set(3,test);
  test = !rpcFwdFlag && !rpcBarFlag && !dtFlag && cscFlag;   // csc event
  m_gmtfilter.set(4,test);
  test = !rpcFwdFlag && rpcBarFlag  && dtFlag  && !cscFlag;  // rpc & dt  event
  m_gmtfilter.set(5,test);
  test = rpcFwdFlag  && !rpcBarFlag && !dtFlag && cscFlag;   // rpc & csc event
  m_gmtfilter.set(6,test);
  test = !rpcFwdFlag && !rpcBarFlag && dtFlag  && cscFlag;   // dt  & csc event
  m_gmtfilter.set(7,test);
  test = (rpcFwdFlag || rpcBarFlag) && dtFlag  && cscFlag;   // rpc & dt & csc event

  //.
  makeGMTFilterDist( );
    
  return nmuons;
  
}

void RPCTechnicalTrigger::makeGMTFilterDist( )
{
  
  //.
  if ( !m_gmtfilter.any() ) return;
  
  for(int i=0; i< 8; ++i){
    if ( m_gmtfilter[i] ) {
      
      m_houtput->htrigdecision->Fill( (2*i)+ 1 );
      
      if ( m_cand < 100 )
        m_houtput->hhtrgperevent->Fill( m_cand, (2*i)+ 1 );
      
    }
    
  }
  
}

DEFINE_FWK_MODULE(RPCTechnicalTrigger);

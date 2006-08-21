//---------------------------------------------
//
//   \class L1MuGlobalMuonTrigger
//
//   Description: L1 Global Muon Trigger
//
//
//   $Date: 2006/05/15 13:56:02 $
//   $Revision: 1.1 $
//
//   Author :
//   Norbert Neumeister              CERN EP
//   Hannes Sakulin                  HEPHY Vienna
//   Ivan Mikulec                    HEPHY Vienna
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPSB.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatcher.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTCancelOutUnit.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMipIsoAU.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMerger.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTSorter.h"

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//----------------
// Constructors --
//----------------
L1MuGlobalMuonTrigger::L1MuGlobalMuonTrigger(const edm::ParameterSet& ps) {
  produces<std::vector<L1MuGMTCand> >();
  produces<L1MuGMTReadoutCollection>();

  m_ExtendedCands.reserve(20);

  // set configuration parameters
  if(!m_config) m_config = new L1MuGMTConfig(ps);

  // build GMT
  if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_info") << endl;
  if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_info") << "**** L1GlobalMuonTrigger building ****"
                                      << endl;
  if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_info") << endl;

  // create new PSB
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT PSB" << endl;
  m_PSB = new L1MuGMTPSB(*this);

  // create new matcher
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Matcher (0,1)" << endl;
  m_Matcher[0] = new L1MuGMTMatcher(*this,0);   // barrel
  m_Matcher[1] = new L1MuGMTMatcher(*this,1);   // endcap

  // create new cancel-out units
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Cancel Out Unit (0,1,2,3)" << endl;
  m_CancelOutUnit[0] = new L1MuGMTCancelOutUnit(*this,0);   // barrel
  m_CancelOutUnit[1] = new L1MuGMTCancelOutUnit(*this,1);   // endcap
  m_CancelOutUnit[2] = new L1MuGMTCancelOutUnit(*this,2);   // CSC/bRPC
  m_CancelOutUnit[3] = new L1MuGMTCancelOutUnit(*this,3);   // DT/fRPC

  // create new MIP & ISO bit assignment units
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT MIP & ISO bit Assigment Unit (0,1)" << endl;
  m_MipIsoAU[0] = new L1MuGMTMipIsoAU(*this,0);   // barrel
  m_MipIsoAU[1] = new L1MuGMTMipIsoAU(*this,1);   // endcap

  // create new Merger
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Merger (0,1)" << endl;
  m_Merger[0] = new L1MuGMTMerger(*this,0);   // barrel
  m_Merger[1] = new L1MuGMTMerger(*this,1);   // endcap

  // create new sorter
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Sorter" << endl;
  m_Sorter = new L1MuGMTSorter(*this);   // barrel

  if(!m_db) m_db = new L1MuGMTDebugBlock(m_config->getBxMin(),m_config->getBxMax());
}

//--------------
// Destructor --
//--------------
L1MuGlobalMuonTrigger::~L1MuGlobalMuonTrigger() {

  if(m_db) delete m_db;
  m_db = 0;

  delete m_Sorter;
  delete m_Merger[1];        // endcap Merger
  delete m_Merger[0];        // barrel Merger
  delete m_MipIsoAU[1];      // barrel MIP & ISO bit assignment unit
  delete m_MipIsoAU[0];      // barrel MIP & ISO bit assignment unit
  delete m_CancelOutUnit[3]; // DT/fRPC cancel-out unit (in endcap chip)
  delete m_CancelOutUnit[2]; // CSC/bRPC cancel-out unit (in barrel chip)
  delete m_CancelOutUnit[1]; // endcap DT/CSC cancel out unit
  delete m_CancelOutUnit[0]; // barrel DT/CSC cancel out unit
  delete m_Matcher[1];       // endcap matcher
  delete m_Matcher[0];       // barrel matcher
  delete m_PSB;

  if(m_config) delete m_config;
  m_config = 0;

}

//--------------
// Operations --
//--------------

void L1MuGlobalMuonTrigger::produce(edm::Event& e, const edm::EventSetup& es) {
  
  // process the event
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << endl;
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "**** L1GlobalMuonTrigger processing ****"
                                      << endl;
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << endl;

  int bx_min = L1MuGMTConfig::getBxMin();
  int bx_max = L1MuGMTConfig::getBxMax();

  m_ExtendedCands.clear();

  // clear readout ring buffer
  vector<L1MuGMTReadoutRecord*>::iterator irr = m_ReadoutRingbuffer.begin();
  for ( ;irr!=m_ReadoutRingbuffer.end(); irr++) delete (*irr);
  m_ReadoutRingbuffer.clear();

  if(m_db) m_db->reset(); // reset debug block

  for ( int bx = bx_min; bx <= bx_max; bx++ ) {
    m_db->SetBX(bx);

    // create new element in readout ring buffer
    m_ReadoutRingbuffer.push_back( new L1MuGMTReadoutRecord(bx) );

    if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "L1GlobalMuonTrigger processing bunch-crossing : " << bx << endl;

    // get data into the data buffer
    if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT PSB" << endl;
    if ( m_PSB ) {
      m_PSB->receiveData(e,bx);
      if ( L1MuGMTConfig::Debug(4) ) m_PSB->print();
    }

    if ( m_PSB && !m_PSB->empty() ) {

      // run matcher
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel Matcher" << endl;
      if ( m_Matcher[0] ) m_Matcher[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Matcher[0] ) m_Matcher[0]->print();
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap Matcher" << endl;
      if ( m_Matcher[1] ) m_Matcher[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Matcher[1] ) m_Matcher[1]->print();

      // run cancel-out units
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel Cancel Out Unit" << endl;
      if ( m_CancelOutUnit[0] ) m_CancelOutUnit[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[0] ) m_CancelOutUnit[0]->print();

      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap Cancel Out Unit" << endl;
      if ( m_CancelOutUnit[1] ) m_CancelOutUnit[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[1] ) m_CancelOutUnit[1]->print();

      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT CSC/fRPC Cancel Out Unit" << endl;
      if ( m_CancelOutUnit[2] ) m_CancelOutUnit[2]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[2] ) m_CancelOutUnit[2]->print();

      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT DT/bRPC Cancel Out Unit" << endl;
      if ( m_CancelOutUnit[3] ) m_CancelOutUnit[3]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[3] ) m_CancelOutUnit[3]->print();

      // run MIP & ISO bit assignment units
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel MIP & ISO bit Assignment Unit" << endl;
      if ( m_MipIsoAU[0] ) m_MipIsoAU[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_MipIsoAU[0] ) m_MipIsoAU[0]->print();
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap MIP & ISO bit Assignment Unit" << endl;
      if ( m_MipIsoAU[1] ) m_MipIsoAU[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_MipIsoAU[1] ) m_MipIsoAU[1]->print();

      // run Merger
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel Merger" << endl;
      if ( m_Merger[0] ) m_Merger[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Merger[0] ) m_Merger[0]->print();
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap Merger" << endl;
      if ( m_Merger[1] ) m_Merger[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Merger[1] ) m_Merger[1]->print();

      // run sorter
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT Sorter" << endl;
      if ( m_Sorter ) m_Sorter->run();
      if ( L1MuGMTConfig::Debug(1) && m_Sorter ) m_Sorter->print();

      // store found track candidates in a container
      if ( m_Sorter->numberOfCands() > 0 ) {
        const vector<const L1MuGMTExtendedCand*>&  gmt_cont = m_Sorter->Cands();
        vector<const L1MuGMTExtendedCand*>::const_iterator iexc;
        for ( iexc = gmt_cont.begin(); iexc != gmt_cont.end(); iexc++ ) {
          if ( *iexc ) m_ExtendedCands.push_back( **iexc );
        }
      }

      // reset GMT
      reset();

    }
  }

  // produce the output
  auto_ptr<vector<L1MuGMTCand> > GMTCands(new vector<L1MuGMTCand>);
  vector<L1MuGMTExtendedCand>::const_iterator iexc;
  for(iexc=m_ExtendedCands.begin(); iexc!=m_ExtendedCands.end(); iexc++) {
    GMTCands->push_back(*iexc);
  }
  e.put(GMTCands);

  auto_ptr<L1MuGMTReadoutCollection> GMTRRC(getReadoutCollection());
  e.put(GMTRRC);
  
}

//
// reset GMT
//
void L1MuGlobalMuonTrigger::reset() {

  if ( m_PSB ) m_PSB->reset();
  if ( m_Matcher[0] ) m_Matcher[0]->reset();
  if ( m_Matcher[1] ) m_Matcher[1]->reset();
  if ( m_CancelOutUnit[0] ) m_CancelOutUnit[0]->reset();
  if ( m_CancelOutUnit[1] ) m_CancelOutUnit[1]->reset();
  if ( m_CancelOutUnit[2] ) m_CancelOutUnit[2]->reset();
  if ( m_CancelOutUnit[3] ) m_CancelOutUnit[3]->reset();
  if ( m_MipIsoAU[0] ) m_MipIsoAU[0]->reset();
  if ( m_MipIsoAU[1] ) m_MipIsoAU[1]->reset();
  if ( m_Merger[0] ) m_Merger[0]->reset();
  if ( m_Merger[1] ) m_Merger[1]->reset();
  if ( m_Sorter) m_Sorter->reset();

}

// get the GMT readout data for the triggered bx
auto_ptr<L1MuGMTReadoutCollection> L1MuGlobalMuonTrigger::getReadoutCollection() {

  int bx_min_ro = L1MuGMTConfig::getBxMinRo();
  int bx_max_ro = L1MuGMTConfig::getBxMaxRo();
  int bx_size = bx_max_ro - bx_min_ro + 1;

  auto_ptr<L1MuGMTReadoutCollection> rrc(new L1MuGMTReadoutCollection(bx_size));

  for (int bx = bx_min_ro; bx <= bx_max_ro; bx++) {
    vector<L1MuGMTReadoutRecord*>::const_iterator iter = m_ReadoutRingbuffer.begin();

    for ( ;iter!=m_ReadoutRingbuffer.end(); iter++) {

      if ( (*iter)->getBxCounter() == bx ) {
        rrc->addRecord(**iter);
        break;
      }
    }

  }

  return rrc;
}

// static data members

L1MuGMTConfig* L1MuGlobalMuonTrigger::m_config = 0;
L1MuGMTDebugBlock* L1MuGlobalMuonTrigger::m_db = 0;

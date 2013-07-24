//---------------------------------------------
//
//   \class L1MuGlobalMuonTrigger
//
//   Description: L1 Global Muon Trigger
//
//
//   $Date: 2010/02/12 12:07:37 $
//   $Revision: 1.14 $
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
#include <sys/stat.h>
#include <sys/types.h>

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

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/DataRecord/interface/L1MuGMTScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTParameters.h"
#include "CondFormats/DataRecord/interface/L1MuGMTParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTChannelMask.h"
#include "CondFormats/DataRecord/interface/L1MuGMTChannelMaskRcd.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

//----------------
// Constructors --
//----------------
L1MuGlobalMuonTrigger::L1MuGlobalMuonTrigger(const edm::ParameterSet& ps) {
  produces<std::vector<L1MuGMTCand> >();
  produces<L1MuGMTReadoutCollection>();
  m_sendMipIso = ps.getUntrackedParameter<bool>("SendMipIso",false);
  if( m_sendMipIso ) {
    produces<std::vector<unsigned> >();
  }
  
  m_L1MuGMTScalesCacheID = 0ULL;
  m_L1MuTriggerScalesCacheID = 0ULL;
  m_L1MuTriggerPtScaleCacheID = 0ULL;
  m_L1MuGMTParametersCacheID = 0ULL;
  m_L1MuGMTChannelMaskCacheID = 0ULL;
  m_L1CaloGeometryCacheID = 0ULL;
  
  m_ExtendedCands.reserve(20);

  // set configuration parameters
  if(!m_config) m_config = new L1MuGMTConfig(ps);
  m_writeLUTsAndRegs = ps.getUntrackedParameter<bool>("WriteLUTsAndRegs",false);

  // build GMT
  if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_info");
  if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_info") << "**** L1GlobalMuonTrigger building ****";
  if ( L1MuGMTConfig::Debug(1) ) edm::LogVerbatim("GMT_info");

  // create new PSB
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT PSB";
  m_PSB = new L1MuGMTPSB(*this);

  // create new matcher
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Matcher (0,1)";
  m_Matcher[0] = new L1MuGMTMatcher(*this,0);   // barrel
  m_Matcher[1] = new L1MuGMTMatcher(*this,1);   // endcap

  // create new cancel-out units
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Cancel Out Unit (0,1,2,3)";
  m_CancelOutUnit[0] = new L1MuGMTCancelOutUnit(*this,0);   // barrel
  m_CancelOutUnit[1] = new L1MuGMTCancelOutUnit(*this,1);   // endcap
  m_CancelOutUnit[2] = new L1MuGMTCancelOutUnit(*this,2);   // CSC/bRPC
  m_CancelOutUnit[3] = new L1MuGMTCancelOutUnit(*this,3);   // DT/fRPC

  // create new MIP & ISO bit assignment units
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT MIP & ISO bit Assigment Unit (0,1)";
  m_MipIsoAU[0] = new L1MuGMTMipIsoAU(*this,0);   // barrel
  m_MipIsoAU[1] = new L1MuGMTMipIsoAU(*this,1);   // endcap

  // create new Merger
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Merger (0,1)";
  m_Merger[0] = new L1MuGMTMerger(*this,0);   // barrel
  m_Merger[1] = new L1MuGMTMerger(*this,1);   // endcap

  // create new sorter
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "creating GMT Sorter";
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

  // copied from produce() by Jim B, 7 Aug 2007
  std::vector<L1MuGMTReadoutRecord*>::iterator irr = m_ReadoutRingbuffer.begin();
  for ( ;irr!=m_ReadoutRingbuffer.end(); irr++) delete (*irr);
  m_ReadoutRingbuffer.clear();
  // end Jim B edit

}

//--------------
// Operations --
//--------------

void L1MuGlobalMuonTrigger::beginJob() {
    
}

void L1MuGlobalMuonTrigger::produce(edm::Event& e, const edm::EventSetup& es) {
  
  // configure from the event setup
  
  unsigned long long L1MuGMTScalesCacheID = es.get< L1MuGMTScalesRcd >().cacheIdentifier();
  if(L1MuGMTScalesCacheID != m_L1MuGMTScalesCacheID) {
    edm::ESHandle< L1MuGMTScales > gmtscales_h;
    es.get< L1MuGMTScalesRcd >().get( gmtscales_h );
    m_config->setGMTScales( gmtscales_h.product() );
  }

  unsigned long long L1MuTriggerScalesCacheID = es.get< L1MuTriggerScalesRcd >().cacheIdentifier();
  if(L1MuTriggerScalesCacheID != m_L1MuTriggerScalesCacheID) {
    edm::ESHandle< L1MuTriggerScales > trigscales_h;
    es.get< L1MuTriggerScalesRcd >().get( trigscales_h );
    m_config->setTriggerScales( trigscales_h.product() );
  }

  unsigned long long L1MuTriggerPtScaleCacheID = es.get< L1MuTriggerPtScaleRcd >().cacheIdentifier();
  if(L1MuTriggerPtScaleCacheID != m_L1MuTriggerPtScaleCacheID) {
    edm::ESHandle< L1MuTriggerPtScale > trigptscale_h;
    es.get< L1MuTriggerPtScaleRcd >().get( trigptscale_h );
    m_config->setTriggerPtScale( trigptscale_h.product() );
  }

  unsigned long long L1MuGMTParametersCacheID = es.get< L1MuGMTParametersRcd >().cacheIdentifier();
  if(L1MuGMTParametersCacheID != m_L1MuGMTParametersCacheID) {
    edm::ESHandle< L1MuGMTParameters > gmtparams_h;
    es.get< L1MuGMTParametersRcd >().get( gmtparams_h );
    m_config->setGMTParams( gmtparams_h.product() );
    m_config->setDefaults();
  }

  unsigned long long L1MuGMTChannelMaskCacheID = es.get< L1MuGMTChannelMaskRcd >().cacheIdentifier();
  if(L1MuGMTChannelMaskCacheID != m_L1MuGMTChannelMaskCacheID) {
    edm::ESHandle< L1MuGMTChannelMask > gmtchanmask_h;
    es.get< L1MuGMTChannelMaskRcd >().get( gmtchanmask_h );
    m_config->setGMTChanMask( gmtchanmask_h.product() );
    if ( L1MuGMTConfig::Debug(1) ) {
      std::string onoff;
      const L1MuGMTChannelMask* theChannelMask = L1MuGMTConfig::getGMTChanMask();
      unsigned mask = theChannelMask->getSubsystemMask();
      
      edm::LogVerbatim("GMT_info");
      edm::LogVerbatim("GMT_info") << " GMT input Channel Mask:" << std::hex << mask << std::dec;
      onoff = mask&1 ? "OFF" : "ON";
      edm::LogVerbatim("GMT_info") << " DT   input " << onoff;
      onoff = mask&2 ? "OFF" : "ON";
      edm::LogVerbatim("GMT_info") << " RPCb input " << onoff;
      onoff = mask&4 ? "OFF" : "ON";
      edm::LogVerbatim("GMT_info") << " CSC  input " << onoff;
      onoff = mask&8 ? "OFF" : "ON";
      edm::LogVerbatim("GMT_info") << " RPCf input " << onoff;
      edm::LogVerbatim("GMT_info");
    }
  }

  unsigned long long L1CaloGeometryCacheID = es.get< L1CaloGeometryRecord >().cacheIdentifier();
  if(L1CaloGeometryCacheID != m_L1CaloGeometryCacheID) {
    edm::ESHandle< L1CaloGeometry > caloGeom_h ;
    es.get< L1CaloGeometryRecord >().get( caloGeom_h ) ;
    m_config->setCaloGeom( caloGeom_h.product() ) ;
  }
  
  m_config->createLUTsRegs();
  
  // write LUTs and Regs if required
  
  if(m_writeLUTsAndRegs) {
    std::string dir = "gmtconfig";
  
    mkdir(dir.c_str(), S_ISUID|S_ISGID|S_ISVTX|S_IRUSR|S_IWUSR|S_IXUSR);

    m_config->dumpLUTs(dir);
    m_config->dumpRegs(dir);
  }
  
  // process the event
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info");
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "**** L1GlobalMuonTrigger processing ****";
  if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info");

  int bx_min = L1MuGMTConfig::getBxMin();
  int bx_max = L1MuGMTConfig::getBxMax();

  m_ExtendedCands.clear();

  // clear readout ring buffer
  std::vector<L1MuGMTReadoutRecord*>::iterator irr = m_ReadoutRingbuffer.begin();
  for ( ;irr!=m_ReadoutRingbuffer.end(); irr++) delete (*irr);
  m_ReadoutRingbuffer.clear();

  if(m_db) m_db->reset(); // reset debug block

  for ( int bx = bx_min; bx <= bx_max; bx++ ) {
    m_db->SetBX(bx);

    // create new element in readout ring buffer
    m_ReadoutRingbuffer.push_back( new L1MuGMTReadoutRecord(bx) );

    if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "L1GlobalMuonTrigger processing bunch-crossing : " << bx;

    // get data into the data buffer
    if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT PSB";
    if ( m_PSB ) {
      m_PSB->receiveData(e,bx);
      if ( L1MuGMTConfig::Debug(4) ) m_PSB->print();
    }

    if ( m_PSB && !m_PSB->empty() ) {

      // run matcher
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel Matcher";
      if ( m_Matcher[0] ) m_Matcher[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Matcher[0] ) m_Matcher[0]->print();
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap Matcher";
      if ( m_Matcher[1] ) m_Matcher[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Matcher[1] ) m_Matcher[1]->print();

      // run cancel-out units
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel Cancel Out Unit";
      if ( m_CancelOutUnit[0] ) m_CancelOutUnit[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[0] ) m_CancelOutUnit[0]->print();

      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap Cancel Out Unit";
      if ( m_CancelOutUnit[1] ) m_CancelOutUnit[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[1] ) m_CancelOutUnit[1]->print();

      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT CSC/fRPC Cancel Out Unit";
      if ( m_CancelOutUnit[2] ) m_CancelOutUnit[2]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[2] ) m_CancelOutUnit[2]->print();

      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT DT/bRPC Cancel Out Unit";
      if ( m_CancelOutUnit[3] ) m_CancelOutUnit[3]->run();
      if ( L1MuGMTConfig::Debug(3) && m_CancelOutUnit[3] ) m_CancelOutUnit[3]->print();

      // run MIP & ISO bit assignment units
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel MIP & ISO bit Assignment Unit";
      if ( m_MipIsoAU[0] ) m_MipIsoAU[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_MipIsoAU[0] ) m_MipIsoAU[0]->print();
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap MIP & ISO bit Assignment Unit";
      if ( m_MipIsoAU[1] ) m_MipIsoAU[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_MipIsoAU[1] ) m_MipIsoAU[1]->print();

      // run Merger
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT barrel Merger";
      if ( m_Merger[0] ) m_Merger[0]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Merger[0] ) m_Merger[0]->print();
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT endcap Merger";
      if ( m_Merger[1] ) m_Merger[1]->run();
      if ( L1MuGMTConfig::Debug(3) && m_Merger[1] ) m_Merger[1]->print();

      // run sorter
      if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("GMT_info") << "running GMT Sorter";
      if ( m_Sorter ) m_Sorter->run();
      if ( L1MuGMTConfig::Debug(1) && m_Sorter ) m_Sorter->print();

      // store found track candidates in a container
      if ( m_Sorter->numberOfCands() > 0 ) {
        const std::vector<const L1MuGMTExtendedCand*>&  gmt_cont = m_Sorter->Cands();
        std::vector<const L1MuGMTExtendedCand*>::const_iterator iexc;
        for ( iexc = gmt_cont.begin(); iexc != gmt_cont.end(); iexc++ ) {
          if ( *iexc ) m_ExtendedCands.push_back( **iexc );
        }
      }

      // reset GMT
      reset();

    }
  }

  // produce the output
  std::auto_ptr<std::vector<L1MuGMTCand> > GMTCands(new std::vector<L1MuGMTCand>);
  std::vector<L1MuGMTExtendedCand>::const_iterator iexc;
  for(iexc=m_ExtendedCands.begin(); iexc!=m_ExtendedCands.end(); iexc++) {
    GMTCands->push_back(*iexc);
  }
  e.put(GMTCands);

  std::auto_ptr<L1MuGMTReadoutCollection> GMTRRC(getReadoutCollection());
  e.put(GMTRRC);

  if( m_sendMipIso ) {
    std::auto_ptr<std::vector<unsigned> > mipiso(new std::vector<unsigned>);
    for(int i=0; i<32; i++) {
      mipiso->push_back(m_db->IsMIPISO(0,i));
    }  
    e.put(mipiso);
  }
  
// delete registers and LUTs
  m_config->clearLUTsRegs();
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
std::auto_ptr<L1MuGMTReadoutCollection> L1MuGlobalMuonTrigger::getReadoutCollection() {

  int bx_min_ro = L1MuGMTConfig::getBxMinRo();
  int bx_max_ro = L1MuGMTConfig::getBxMaxRo();
  int bx_size = bx_max_ro - bx_min_ro + 1;

  std::auto_ptr<L1MuGMTReadoutCollection> rrc(new L1MuGMTReadoutCollection(bx_size));

  for (int bx = bx_min_ro; bx <= bx_max_ro; bx++) {
    std::vector<L1MuGMTReadoutRecord*>::const_iterator iter = m_ReadoutRingbuffer.begin();

    for ( ;iter!=m_ReadoutRingbuffer.end(); iter++) {

      if ( (*iter)->getBxInEvent() == bx ) {
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

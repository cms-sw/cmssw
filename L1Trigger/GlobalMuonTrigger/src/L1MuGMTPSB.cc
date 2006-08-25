//-------------------------------------------------
//
//   Class: L1MuGMTPSB
//
//   Description: Pipelined Synchronising Buffer module 
//
//
//   $Date: 2006/08/21 14:23:13 $
//   $Revision: 1.2 $
//
//   Author :
//   N. Neumeister            CERN EP 
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPSB.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <vector>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"

#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTPSB
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTPSB::L1MuGMTPSB(const L1MuGlobalMuonTrigger& gmt) : 
               m_gmt(gmt), 
	       m_RpcMuons(L1MuGMTConfig::MAXRPC),
               m_DtbxMuons(L1MuGMTConfig::MAXDTBX), 
               m_CscMuons(L1MuGMTConfig::MAXCSC),
               m_Isol(14,18), m_Mip(14,18) {

  m_RpcMuons.reserve(L1MuGMTConfig::MAXRPC);
  m_DtbxMuons.reserve(L1MuGMTConfig::MAXDTBX);
  m_CscMuons.reserve(L1MuGMTConfig::MAXCSC);

}

//--------------
// Destructor --
//--------------
L1MuGMTPSB::~L1MuGMTPSB() { 

  reset();
  m_RpcMuons.clear();
  m_DtbxMuons.clear();
  m_CscMuons.clear();
  
}

//--------------
// Operations --
//--------------

//
// receive data
//
void L1MuGMTPSB::receiveData(edm::Event& e, int bx) {
  ////////////////////////////////////////

  std::vector<edm::Handle<std::vector<L1MuRegionalCand> > > alldata;
  e.getManyByType(alldata);

  std::vector<edm::Handle<std::vector<L1MuRegionalCand> > >::iterator it;
  for(it=alldata.begin(); it!=alldata.end(); it++) {
    edm::Provenance const* prov = it->provenance();
    if(prov->product.productInstanceName() == "DT")  getDTBX(it->product(),bx);
    if(prov->product.productInstanceName() == "dt")  getDTBX(it->product(),bx);
    if(prov->product.productInstanceName() == "CSC")  getCSC(it->product(),bx);
    if(prov->product.productInstanceName() == "RPCb")  getRPCb(it->product(),bx);
    if(prov->product.productInstanceName() == "RPCf")  getRPCf(it->product(),bx);
  }

  ////////////////////////////////////

  // store data in readout record
  for (int i=0; i<4; i++)
    m_gmt.currentReadoutRecord()->setInputCand ( i, DTBXMuon(i)->getDataWord() );
  for (int i=0; i<4; i++)
    m_gmt.currentReadoutRecord()->setInputCand ( i+4, RPCMuon(i)->getDataWord() );
  for (int i=0; i<4; i++)
    m_gmt.currentReadoutRecord()->setInputCand ( i+8, CSCMuon(i)->getDataWord() );
  for (int i=0; i<4; i++)
    m_gmt.currentReadoutRecord()->setInputCand ( i+12, RPCMuon(i+4)->getDataWord() );


  // if there is at least one muon start the calorimeter trigger 
  //  if ( L1MuGMTConfig::getCaloTrigger() && !empty() ) getCalo(e);

}


//
// clear PSB
//
void L1MuGMTPSB::reset() {

  vector<L1MuRegionalCand>::iterator iter; 
  iter = m_RpcMuons.begin();
  while ( iter != m_RpcMuons.end() ) (*(iter++)).reset();

  iter = m_DtbxMuons.begin();
  while ( iter != m_DtbxMuons.end() )(*(iter++)).reset();

  iter = m_CscMuons.begin();
  while ( iter != m_CscMuons.end() ) (*(iter++)).reset();
  
  m_Isol.init(false);
  m_Mip.init(false);

}


//
// print muons
//
void L1MuGMTPSB::print() const {

  edm::LogVerbatim("GMT_PSB_info") << endl;
  printDTBX();
  printRPCbarrel();
  printCSC();
  printRPCendcap();
  edm::LogVerbatim("GMT_PSB_info") << endl;
  
}


//
// return RPC muon
//
const L1MuRegionalCand* L1MuGMTPSB::RPCMuon(int index) const {
  
  return ( index < (int)L1MuGMTConfig::MAXRPC && index >= 0 ) ? &(m_RpcMuons[index]) : 0;
  
}


//
// return DTBX muon
//
const L1MuRegionalCand* L1MuGMTPSB::DTBXMuon(int index) const {
      
  return ( index < (int)L1MuGMTConfig::MAXDTBX && index >= 0 ) ? &(m_DtbxMuons[index]) : 0;
  
}


//
// return CSC muon
//
const L1MuRegionalCand* L1MuGMTPSB::CSCMuon(int index) const {
 
  return ( index < (int)L1MuGMTConfig::MAXCSC && index >= 0 ) ? &(m_CscMuons[index]) : 0;
  
}

//
// count number of non empty RPC muons
//
int L1MuGMTPSB::numberRPC() const {

  int count = 0;
  vector<L1MuRegionalCand>::const_iterator iter = m_RpcMuons.begin();
  while ( iter != m_RpcMuons.end() ) {
    if ( !(*iter).empty() ) count++;
    iter++;
  }
  return count;

}


//
// count number of non empty DT muons
//
int L1MuGMTPSB::numberDTBX() const {

  int count = 0;
  vector<L1MuRegionalCand>::const_iterator iter = m_DtbxMuons.begin();
  while ( iter != m_DtbxMuons.end() ) {
    if ( !(*iter).empty() ) count++;
    iter++;
  }
  return count;

}


//
// count number of non empty CSC muons
//
int L1MuGMTPSB::numberCSC() const {

  int count = 0;
  vector<L1MuRegionalCand>::const_iterator iter = m_CscMuons.begin();
  while ( iter != m_CscMuons.end() ) {
    if ( !(*iter).empty() ) count++;
    iter++;
  }
  return count;

}


//
// are there any data in the PSB
//
bool L1MuGMTPSB::empty() const {

  int number = numberRPC() + numberDTBX() + numberCSC();

  return ( number == 0 );

}


//
// get muons from RPCb Trigger
//
void L1MuGMTPSB::getRPCb(std::vector<L1MuRegionalCand> const* data, int bx) {

  int irpcb = 0;
  std::vector<L1MuRegionalCand>::const_iterator iter;
  for ( iter = data->begin(); iter != data->end(); iter++ ) {
    if ( (*iter).bx() != bx ) continue;
    if ( irpcb < (int)L1MuGMTConfig::MAXRPCbarrel ) { 
      if(!(*iter).empty()) m_RpcMuons[irpcb] = (*iter);
      irpcb++;
    }  
  }
  
}


//
// get muons from RPCf Trigger
//
void L1MuGMTPSB::getRPCf(std::vector<L1MuRegionalCand> const* data, int bx) {

  int irpcf = 0;
  std::vector<L1MuRegionalCand>::const_iterator iter;
  for ( iter = data->begin(); iter != data->end(); iter++ ) {
    if ( (*iter).bx() != bx ) continue;
    if ( irpcf < (int)L1MuGMTConfig::MAXRPCendcap ) { 
      if(!(*iter).empty()) m_RpcMuons[irpcf+4] = (*iter);
      irpcf++;
    }  
  }
  
}


//
// get muons from barrel Muon Trigger Track Finder
//
void L1MuGMTPSB::getDTBX(std::vector<L1MuRegionalCand> const* data, int bx) {

  // temporary hack with bxoffset - to be removed, trigger bx should be 0
  int bxoffset = 0;
  int idtbx = 0;
  std::vector<L1MuRegionalCand>::const_iterator iter;
  for ( iter = data->begin(); iter != data->end(); iter++ ) {
    if ( L1MuGMTConfig::Debug(2) ) edm::LogVerbatim("") << "DTTF BX: " << (*iter).bx() << " my bx: " << bx << endl;
    if ( (*iter).bx() > 10) bxoffset=16;
    if ( (*iter).bx() != bx+bxoffset ) continue;
    if ( idtbx < (int)L1MuGMTConfig::MAXDTBX ) { 
      m_DtbxMuons[idtbx] = (*iter);
      m_DtbxMuons[idtbx].setBx(bx);
      idtbx++;
    }  
  }

}


//
// get muons from CSC Track Finder
//
void L1MuGMTPSB::getCSC(std::vector<L1MuRegionalCand> const* data, int bx) {

  int icsc = 0;
  std::vector<L1MuRegionalCand>::const_iterator iter;
  for ( iter = data->begin(); iter != data->end(); iter++ ) {
    if ( (*iter).bx() != bx ) continue;
    if ( icsc < (int)L1MuGMTConfig::MAXCSC ) { 
      m_CscMuons[icsc] = (*iter);
      icsc++;
    }
  }

}


//
// print barrel RPC muons
//
void L1MuGMTPSB::printRPCbarrel() const {

  edm::LogVerbatim("GMT_PSB_info") << "RPC barrel  muons received by the GMT :" << endl;

  for ( unsigned i = 0; i < L1MuGMTConfig::MAXRPCbarrel; i++ ) {
    if (!m_RpcMuons[i].empty()) m_RpcMuons[i].print();
  }
  
}


//
// print endcap RPC muons
//
void L1MuGMTPSB::printRPCendcap() const {

  edm::LogVerbatim("GMT_PSB_info") << "RPC endcap  muons received by the GMT :" << endl;

  for ( unsigned i = 0; i < L1MuGMTConfig::MAXRPCendcap; i++ ) {
    if (!m_RpcMuons[i+4].empty()) m_RpcMuons[i+4].print();
  }

}


//
// print DTBX muons
//
void L1MuGMTPSB::printDTBX() const {

  edm::LogVerbatim("GMT_PSB_info") << "DTBX muons received by the GMT :" << endl;

  for ( unsigned i = 0; i < L1MuGMTConfig::MAXDTBX; i++ ) {
    if (!m_DtbxMuons[i].empty()) m_DtbxMuons[i].print();
  }
  
}


//
// print CSC muons
//
void L1MuGMTPSB::printCSC() const {

  edm::LogVerbatim("GMT_PSB_info") << "CSC  muons received by the GMT :" << endl;

  for ( unsigned i = 0; i < L1MuGMTConfig::MAXCSC; i++ ) {
    if (!m_CscMuons[i].empty()) m_CscMuons[i].print();
  }

}


//
// get data from regional calorimeter trigger
//
void L1MuGMTPSB::getCalo() {
  
  /*
  L1CaloTriggerSetup* setup = Singleton<L1CaloTriggerSetup>::instance();
  L1CRegions* regions = setup->Regions();

  L1CaloTriggerSetup::Regions_const_iter iter;
  for ( iter = regions->begin(); iter != regions->end(); iter++ ) {
    if ( (*iter).eta() < 4 || (*iter).eta() > 17 || (*iter).phi() > 17 ) continue;
    m_Isol.set( (*iter).eta()-4, (*iter).phi(), (*iter).isItQuiet() );
    m_Mip.set( (*iter).eta()-4, (*iter).phi(), (*iter).isItMinI() );

    if ( (*iter).isItQuiet() )
      m_gmt.currentReadoutRecord()->setQuietbit ((*iter).eta()-4, (*iter).phi());

    if ( (*iter).isItMinI() )
      m_gmt.currentReadoutRecord()->setMIPbit ((*iter).eta()-4, (*iter).phi());

  }
  */
}

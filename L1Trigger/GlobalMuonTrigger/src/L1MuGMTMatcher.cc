//-------------------------------------------------
//
//   Class: L1MuGMTMatcher
//
//   Description: Matcher 
//
//
//   $Date: 2007/04/10 09:59:19 $
//   $Revision: 1.4 $
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

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatcher.h"

//---------------
// C++ Headers --
//---------------

#include <iostream>
#include <iomanip>
#include <cmath>

#include <fstream>
#include <sstream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "L1Trigger/GlobalMuonTrigger/interface/L1MuGlobalMuonTrigger.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTPSB.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTDebugBlock.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTSortRankUnit.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFDeltaEtaLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFCOUDeltaEtaLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFOvlEtaConvLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMatchQualLUT.h"
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFDisableHotLUT.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// --------------------------------
//       class L1MuGMTMatcher
//---------------------------------

//----------------
// Constructors --
//----------------
L1MuGMTMatcher::L1MuGMTMatcher(const L1MuGlobalMuonTrigger& gmt, int id) : 
               m_gmt(gmt), m_id(id), first(MaxMatch), second(MaxMatch), 
               matchQuality(MaxMatch,MaxMatch), pairMatrix(MaxMatch,MaxMatch) {

  first.reserve(MaxMatch);
  second.reserve(MaxMatch);

}

//--------------
// Destructor --
//--------------
L1MuGMTMatcher::~L1MuGMTMatcher() { 

  reset();
  
}

//--------------
// Operations --
//--------------

//
// run Muon Matcher
//
void L1MuGMTMatcher::run() {

  load();
  match();
  
}

//
// clear Matcher
//
void L1MuGMTMatcher::reset() {

  matchQuality.init(0);
  pairMatrix.init(false);
  
  for ( unsigned i = 0; i < MaxMatch; i++ ) {
    first[i] = 0;
    second[i] = 0;
  }  

}


//
// print matching results
//
void L1MuGMTMatcher::print() {

  edm::LogVerbatim("GMT_Matcher_info");
  if ( L1MuGMTConfig::Debug(4) ) {
    edm::LogVerbatim("GMT_Matcher_info") << "MATCH Quality : ";
    matchQuality.print();
  }

  edm::LogVerbatim("GMT_Matcher_info") << "PAIR Matrix : ";
  pairMatrix.print();

  edm::LogVerbatim("GMT_Matcher_info");

}


//
// load Matcher (get data from data buffer)
//
void L1MuGMTMatcher::load() {

  // barrel matcher gets DTBX and barrel RPC muons
  if ( m_id == 0 ) {
    for ( unsigned idt = 0; idt < L1MuGMTConfig::MAXDTBX; idt++ ) {
      first[idt] = m_gmt.Data()->DTBXMuon(idt);
    }
    for ( unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCbarrel; irpc++ ) {
      second[irpc] = m_gmt.Data()->RPCMuon(irpc);
    }
  }
  
  // endcap matcher gets CSC and endcap RPC muons
  if ( m_id == 1 ) {  
    for ( unsigned icsc = 0; icsc < L1MuGMTConfig::MAXCSC; icsc++ ) {
      first[icsc] = m_gmt.Data()->CSCMuon(icsc);
    }
    for ( unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCendcap; irpc++ ) {
      second[irpc] = m_gmt.Data()->RPCMuon(irpc+4);
    }
  }

  // matcher in DT/CSC cancel-out unit gets DTBX and CSC muons
  if ( m_id == 2 ) {
    for ( unsigned idt = 0; idt < L1MuGMTConfig::MAXDTBX; idt++ ) {
      first[idt] = m_gmt.Data()->DTBXMuon(idt);
    }
    for ( unsigned icsc = 0; icsc < L1MuGMTConfig::MAXCSC; icsc++ ) {
      second[icsc] = m_gmt.Data()->CSCMuon(icsc);
    }
  }

  // matcher in CSC/DT cancel-out unit gets CSC and DTBX muons
  if ( m_id==3 ) {
    for ( unsigned icsc = 0; icsc < L1MuGMTConfig::MAXCSC; icsc++ ) {
      first[icsc] = m_gmt.Data()->CSCMuon(icsc);
    }
    for ( unsigned idt = 0; idt < L1MuGMTConfig::MAXDTBX; idt++ ) {
      second[idt] = m_gmt.Data()->DTBXMuon(idt);
    }
  }

  // bRPC/CSC gets barrel RPC and CSC muons
  if ( m_id == 4 ) {  
    for ( unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCbarrel; irpc++ ) {
      first[irpc] = m_gmt.Data()->RPCMuon(irpc);
    }
    for ( unsigned icsc = 0; icsc < L1MuGMTConfig::MAXCSC; icsc++ ) {
      second[icsc] = m_gmt.Data()->CSCMuon(icsc);
    }
  }

  // bRPC/DT matcher gets forward RPC and DTBX muons
  if ( m_id == 5 ) {
    for ( unsigned irpc = 0; irpc < L1MuGMTConfig::MAXRPCendcap; irpc++ ) {
      first[irpc] = m_gmt.Data()->RPCMuon(irpc+4);
    }
    for ( unsigned idt = 0; idt < L1MuGMTConfig::MAXDTBX; idt++ ) {
      second[idt] = m_gmt.Data()->DTBXMuon(idt);
    }
  }
}


//
// match muons
//
void L1MuGMTMatcher::match() {

  L1MuGMTMatrix<bool> maxMatrix(MaxMatch,MaxMatch);
  L1MuGMTMatrix<bool> disableMatrix(MaxMatch,MaxMatch);
  maxMatrix.init(false);
  disableMatrix.init(false);

  // loop over all combinations

  unsigned i,j;
  for ( i = 0; i < MaxMatch; i++ ) 
    for ( j = 0; j < MaxMatch; j++ ) 
      matchQuality(i,j) = lookup_mq(i,j);

  // store in debug block
  m_gmt.DebugBlockForFill()->SetMQMatrix( m_id, matchQuality) ;

  // fill MAX matrix

  for ( i = 0; i < MaxMatch; i++ )
    for ( j = 0; j < MaxMatch; j++ ) 
      maxMatrix(i,j) = matchQuality.isMax(i,j) && (matchQuality(i,j) != 0);
  
  // fill disable matrix
 
  for ( i = 0; i < MaxMatch; i++ )  
    for ( j = 0; j < MaxMatch; j++ ) {

      for ( unsigned i1 = 0; i1 < MaxMatch; i1++ ) 
        if ( i1 != i ) disableMatrix(i,j) = disableMatrix(i,j) || maxMatrix(i1,j);
      
      for ( unsigned j1 = 0; j1 < MaxMatch; j1++ ) 
        if ( j1 != j ) disableMatrix(i,j) = disableMatrix(i,j) || maxMatrix(i,j1);
  }

  // fill pair matrix

  for ( i = 0; i < MaxMatch; i++ ) {  
    for ( j = 0; j < MaxMatch; j++ ) {

      bool max = true;

      for ( unsigned i1 = 0; i1 < i; i1++ ) {
          max = max && ((matchQuality(i,j) > matchQuality(i1,j)) ||
                         disableMatrix(i1,j));
      }

      for ( unsigned i1 = i+1; i1 < MaxMatch; i1++ ) {
          max = max && ((matchQuality(i,j) >= matchQuality(i1,j)) ||
                         disableMatrix(i1,j));
      }

      for ( unsigned j1 = 0; j1 < j; j1++ ) {
          max = max && ((matchQuality(i,j) > matchQuality(i,j1)) ||
                         disableMatrix(i,j1));
      }

      for ( unsigned j1 = j+1; j1 < MaxMatch; j1++ ) {
          max = max && ((matchQuality(i,j) >= matchQuality(i,j1)) ||
                         disableMatrix(i,j1));
      }
 
      pairMatrix(i,j) = max && (matchQuality(i,j) != 0);

    }
  }  

  // store in debug block
  m_gmt.DebugBlockForFill()->SetPairMatrix( m_id, pairMatrix) ;
}

//
// compare eta and phi of two muons
//
int L1MuGMTMatcher::lookup_mq(int i, int j) {

  bool empty1 = ( first[i] != 0 ) ? first[i]->empty() : true;
  bool empty2 = ( second[j] != 0 ) ? second[j]->empty() : true;
  if ( empty1 || empty2) return 0; 

  //
  // (1) calculate delta-phi (integer version)
  //
  unsigned phi1 = first[i]->phi_packed();
  unsigned phi2 = second[j]->phi_packed();
  
  int delta_phi = ( ( phi1 - phi2 + 3*72  ) % 144  ) - 72;
  
  if (delta_phi < -3 || delta_phi >3) 
    delta_phi = -4;
  
  L1MuSignedPacking<3> DPhiPacking;
  unsigned delta_phi_packed = DPhiPacking.packedFromIdx (delta_phi);
  
  //
  // (2) look-up delta-eta
  //
  unsigned eta1 = first[i]->eta_packed();
  unsigned eta2 = second[j]->eta_packed();

  unsigned delta_eta_packed = 0;
  
  if (m_id == 0 || m_id == 1) { // main matching units
    // first is dt/csc, second is rpc

    bool disable1 = L1MuGMTSortRankUnit::isDisabled(first[i]);
    bool disable2 = L1MuGMTSortRankUnit::isDisabled(second[j]);;

    if (disable1 || disable2) return 0;
 
    L1MuGMTLFDeltaEtaLUT* de_lut = L1MuGMTConfig::getLFDeltaEtaLUT(); 
    delta_eta_packed = de_lut->SpecificLookup_delta_eta (m_id, eta1, eta2);
  }
  else { // overlap cancel-out matching units
    // first is own chip, second is other chip
    int idx1 = first[i]->type_idx(); 
    int idx1_dcrr = (idx1==1)?2:(idx1==2)?1:idx1;

    int idx2 = second[j]->type_idx(); 
    int idx2_dcrr = (idx2==1)?2:(idx2==2)?1:idx2;

    bool disable1 = L1MuGMTSortRankUnit::isDisabled(first[i]);
    
    L1MuGMTLFDisableHotLUT* dishot_lut = L1MuGMTConfig::getLFDisableHotLUT(); 
    bool disable2 = dishot_lut->SpecificLookup_disable_hot (idx2_dcrr, 
							    second[j]->eta_packed(), 
							    second[j]->phi_packed()) == 1;

    if (disable1 || disable2) return 0;
    
    // convert eta to 4-bit, first
    L1MuGMTLFOvlEtaConvLUT* econv_lut = L1MuGMTConfig::getLFOvlEtaConvLUT(); 


    unsigned eta1_4bit = econv_lut->SpecificLookup_eta_ovl (idx1_dcrr, eta1);
    unsigned eta2_4bit = econv_lut->SpecificLookup_eta_ovl (idx2_dcrr, eta2);

    // look up delta eta
    L1MuGMTLFCOUDeltaEtaLUT* cou_de_lut = L1MuGMTConfig::getLFCOUDeltaEtaLUT(); 
    delta_eta_packed = cou_de_lut->SpecificLookup_delta_eta (m_id-2, eta1_4bit, eta2_4bit);
  }

  //
  // (3) look up match quality
  //
  L1MuGMTLFMatchQualLUT* mq_lut = L1MuGMTConfig::getLFMatchQualLUT(); 
    
  unsigned mq = mq_lut->SpecificLookup_mq(m_id, delta_eta_packed, delta_phi_packed);

  return mq;
}


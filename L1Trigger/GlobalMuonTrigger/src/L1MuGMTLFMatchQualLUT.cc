//-------------------------------------------------
//
//   Class: L1MuGMTLFMatchQualLUT
//
// 
//   $Date: 2007/04/02 15:45:38 $
//   $Revision: 1.3 $
//
//   Author :
//   H. Sakulin            HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTLFMatchQualLUT.h"

//---------------
// C++ Headers --
//---------------

//#include <iostream>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "CondFormats/L1TObjects/interface/L1MuGMTScales.h"
#include "CondFormats/L1TObjects/interface/L1MuPacking.h"

//-------------------
// InitParameters  --
//-------------------

void L1MuGMTLFMatchQualLUT::InitParameters() {

  m_EtaWeights[0]=L1MuGMTConfig::getEtaWeightBarrel();
  m_PhiWeights[0]=L1MuGMTConfig::getPhiWeightBarrel();
  m_EtaPhiThresholds[0]=L1MuGMTConfig::getEtaPhiThresholdBarrel();
  
  m_EtaWeights[1]=L1MuGMTConfig::getEtaWeightEndcap();
  m_PhiWeights[1]=L1MuGMTConfig::getPhiWeightEndcap();
  m_EtaPhiThresholds[1]=L1MuGMTConfig::getEtaPhiThresholdEndcap();
  
  for (int i=2; i<6; i++) {
    m_EtaWeights[i]=L1MuGMTConfig::getEtaWeightCOU();
    m_PhiWeights[i]=L1MuGMTConfig::getPhiWeightCOU();
    m_EtaPhiThresholds[i]=L1MuGMTConfig::getEtaPhiThresholdCOU();
  }
}

//------------------------
// The Lookup Function  --
//------------------------

unsigned L1MuGMTLFMatchQualLUT::TheLookupFunction (int idx, unsigned delta_eta, unsigned delta_phi) const {
  // idx is DTRPC, CSCRPC, DTCSC, CSCDT, CSCbRPC, DTfRPC
  // INPUTS:  delta_eta(4) delta_phi(3)
  // OUTPUTS: mq(6) 

  const L1MuGMTScales* theGMTScales = L1MuGMTConfig::getGMTScales();

  float deta = theGMTScales->getDeltaEtaScale(idx)->getCenter( delta_eta );
  float dphi = theGMTScales->getDeltaPhiScale()->getCenter( delta_phi );
      
  // check out-of range code
  L1MuSignedPacking<4> EtaPacking;
  int delta_eta_signed = EtaPacking.idxFromPacked(delta_eta);
  L1MuSignedPacking<3> PhiPacking;
  int delta_phi_signed = PhiPacking.idxFromPacked(delta_phi);

  bool dphi_outofrange = (delta_phi_signed == -4);

  // limit delta-phi even further    **FIXME: make this configurable
  if (delta_phi_signed >=3 || delta_phi_signed <=-3) 
    dphi_outofrange = true;

  // check out-of range code
  bool deta_outofrange = (delta_eta_signed == -8);

  // limit delta-eta even further    **FIXME: make this configurable
  if (delta_eta_signed >=7 || delta_eta_signed <=-7) 
    deta_outofrange = true;

  double delta_etaphi = sqrt(m_EtaWeights[idx]*deta*deta + m_PhiWeights[idx]*dphi*dphi);

//    cout << "MQ LUT       : delta_phi = " << dphi
//         << ", delta_eta = " << deta
//         << ", delta_etaphi = " << delta_etaphi
//         << endl;

      
  int matchqual = 0;
  if ( dphi_outofrange || deta_outofrange ||
       delta_etaphi > m_EtaPhiThresholds[idx] ) {
    matchqual = 0;
  }
  else {
    double mq = 64. * (m_EtaPhiThresholds[idx] - delta_etaphi) / m_EtaPhiThresholds[idx];
    matchqual = static_cast<int>(mq);
  }

  if (matchqual > 63) matchqual = 63;
  return matchqual;  
}




















//-------------------------------------------------
//
//   Class L1Phase2MuDTExtPhDigi.cc
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Nicolo' Trevisani - Oviedo ICTEA 
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtPhDigi.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
L1Phase2MuDTExtPhDigi::L1Phase2MuDTExtPhDigi():

  L1Phase2MuDTPhDigi(),

  m_xLocal(0),
  m_tanPsi(0),
  m_phiCMSSW(0),
  m_phiBendCMSSW(0) {
  
  for (int i=0; i<8; i++) {
    m_pathWireId[i] = -1;
    m_pathTDC[i] = -1;
    m_pathLat[i] = 2;
  }
}

L1Phase2MuDTExtPhDigi::L1Phase2MuDTExtPhDigi(int bx, int wh, int sc, int st, int sl, int phi, int phib, int qual, int idx, int t0, int chi2, int x, int tan, int phi_cmssw, int phib_cmssw, int rpc, int wireId[8], int tdc[8],int lat[8]): 

  L1Phase2MuDTPhDigi(bx, wh, sc, st, sl, phi, phib, qual, idx, t0, chi2, rpc),
  
  m_xLocal(x),
  m_tanPsi(tan),
  m_phiCMSSW(phi_cmssw),
  m_phiBendCMSSW(phib_cmssw) {
  
  for (int i=0; i<8; i++) {
    m_pathWireId[i] = wireId[i];
    m_pathTDC[i] = tdc[i];
    m_pathLat[i] = lat[i];
  }
}

L1Phase2MuDTExtPhDigi::L1Phase2MuDTExtPhDigi(const L1Phase2MuDTExtPhDigi &digi):

  L1Phase2MuDTPhDigi(digi.bxNum(), 
		     digi.whNum(), 
		     digi.scNum(), 
		     digi.stNum(), 
		     digi.slNum(), 
		     digi.phi(), 
		     digi.phiBend(), 
		     digi.quality(), 
		     digi.index(), 
		     digi.t0(), 
		     digi.chi2(), 
		     digi.rpcFlag()),

  m_xLocal(digi.xLocal()),
  m_tanPsi(digi.tanPsi()),
  m_phiCMSSW(digi.phiCMSSW()),
  m_phiBendCMSSW(digi.phiBendCMSSW()) {
  
  for (int i=0; i<8; i++) {
    m_pathWireId[i] = digi.pathWireId(i);
    m_pathTDC[i] = digi.pathTDC(i);
    m_pathLat[i] = digi.pathLat(i);
  } 
}

//--------------
// Operations --
//--------------

int L1Phase2MuDTExtPhDigi::xLocal() const { return m_xLocal; }

int L1Phase2MuDTExtPhDigi::tanPsi() const { return m_tanPsi; } 

int L1Phase2MuDTExtPhDigi::phiCMSSW() const { return m_phiCMSSW; } 

int L1Phase2MuDTExtPhDigi::phiBendCMSSW() const { return m_phiBendCMSSW; } 

int L1Phase2MuDTExtPhDigi::pathWireId(int i) const { return m_pathWireId[i]; }

int L1Phase2MuDTExtPhDigi::pathTDC(int i) const { return m_pathTDC[i]; }

int L1Phase2MuDTExtPhDigi::pathLat(int i) const { return m_pathLat[i]; }

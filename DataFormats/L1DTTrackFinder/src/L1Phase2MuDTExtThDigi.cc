//-------------------------------------------------
//
//   Class L1Phase2MuDTExtThDigi.cc
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
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTExtThDigi.h"

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
L1Phase2MuDTExtThDigi::L1Phase2MuDTExtThDigi():

  L1Phase2MuDTThDigi(),

  m_yLocal(0),
  m_zCMSSW(0),
  m_kCMSSW(0) {
  
  for (int i = 0; i < 4; i++) {
    m_pathWireId[i] = -1;
    m_pathTDC[i] = -1;
    m_pathLat[i] = 2;
  }
}

L1Phase2MuDTExtThDigi::L1Phase2MuDTExtThDigi(int bx, int wh, int sc, int st, int z, int k, int qual, int idx, int t0, int chi2, int y, int z_cmssw, int k_cmssw, int rpc, int wireId[4], int tdc[4],int lat[4]): 

  L1Phase2MuDTThDigi(bx, wh, sc, st, z, k, qual, idx, t0, chi2, rpc),
  
  m_yLocal(y),
  m_zCMSSW(z_cmssw),
  m_kCMSSW(k_cmssw) {
  
  for (int i=0; i < 4; i++) {
    m_pathWireId[i] = wireId[i];
    m_pathTDC[i] = tdc[i];
    m_pathLat[i] = lat[i];
  }
}

L1Phase2MuDTExtThDigi::L1Phase2MuDTExtThDigi(const L1Phase2MuDTExtThDigi &digi):

  L1Phase2MuDTThDigi(digi.bxNum(), 
		     digi.whNum(), 
		     digi.scNum(), 
		     digi.stNum(), 
		     digi.z(), 
		     digi.k(), 
		     digi.quality(), 
		     digi.index(), 
		     digi.t0(), 
		     digi.chi2(), 
		     digi.rpcFlag()),

  m_yLocal(digi.yLocal()),
  m_zCMSSW(digi.zCMSSW()),
  m_kCMSSW(digi.kCMSSW()) {
  
  for (int i=0; i < 4; i++) {
    m_pathWireId[i] = digi.pathWireId(i);
    m_pathTDC[i] = digi.pathTDC(i);
    m_pathLat[i] = digi.pathLat(i);
  } 
}

//--------------
// Operations --
//--------------

int L1Phase2MuDTExtThDigi::yLocal() const { return m_yLocal; }

int L1Phase2MuDTExtThDigi::zCMSSW() const { return m_zCMSSW; } 

int L1Phase2MuDTExtThDigi::kCMSSW() const { return m_kCMSSW; } 

int L1Phase2MuDTExtThDigi::pathWireId(int i) const { return m_pathWireId[i]; }

int L1Phase2MuDTExtThDigi::pathTDC(int i) const { return m_pathTDC[i]; }

int L1Phase2MuDTExtThDigi::pathLat(int i) const { return m_pathLat[i]; }

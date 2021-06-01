//-------------------------------------------------
//
//   Class L1Phase2MuDTExtThDigi
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Nicolo' Trevisani - Oviedo
//
//
//--------------------------------------------------
#ifndef L1Phase2MuDTExtThDigi_H
#define L1Phase2MuDTExtThDigi_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTThDigi.h"

//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------

// ---------------------
// -- Class Interface --
// ---------------------

class L1Phase2MuDTExtThDigi : public L1Phase2MuDTThDigi {
 public:
  //  Constructors
  L1Phase2MuDTExtThDigi();

  L1Phase2MuDTExtThDigi(int bx, int wh, int sc, int st, int z, int k, int qual, int idx, int t0, int chi2, int y, int z_cmssw, int k_cmssw, int rpc = -10, int wireId[4] = 0, int tdc[4] = 0,int lat[4] = 0);

  L1Phase2MuDTExtThDigi(const L1Phase2MuDTExtThDigi &digi);
  
  ~L1Phase2MuDTExtThDigi() override {};

  // Operations
  int yLocal() const;

  int zCMSSW() const;
  int kCMSSW() const;

  int pathWireId(int) const;
  int pathTDC(int) const;
  int pathLat(int) const; 

 private:

  int m_yLocal;

  int m_zCMSSW;
  int m_kCMSSW;

  int m_pathWireId[4];
  int m_pathTDC[4];
  int m_pathLat[4];
};

#endif

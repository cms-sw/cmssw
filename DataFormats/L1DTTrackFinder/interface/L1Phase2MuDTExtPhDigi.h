//-------------------------------------------------
//
//   Class L1Phase2MuDTExtPhDigi
//
//   Description: trigger primtive data for the
//                muon barrel Phase2 trigger
//
//
//   Author List: Nicolo' Trevisani - Oviedo ICTEA
//
//
//--------------------------------------------------
#ifndef L1Phase2MuDTExtPhDigi_H
#define L1Phase2MuDTExtPhDigi_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------

// ---------------------
// -- Class Interface --
// ---------------------

class L1Phase2MuDTExtPhDigi : public L1Phase2MuDTPhDigi {
public:
  //  Constructors
  L1Phase2MuDTExtPhDigi();

  L1Phase2MuDTExtPhDigi(int bx,
                        int wh,
                        int sc,
                        int st,
                        int sl,
                        int phi,
                        int phib,
                        int qual,
                        int idx,
                        int t0,
                        int chi2,
                        int x,
                        int tan,
                        int phi_cmssw,
                        int phib_cmssw,
                        int rpc = -10,
                        int wireId[8] = nullptr,
                        int tdc[8] = nullptr,
                        int lat[8] = nullptr);

  L1Phase2MuDTExtPhDigi(const L1Phase2MuDTExtPhDigi &digi);

  ~L1Phase2MuDTExtPhDigi() override{};

  // Operations
  int xLocal() const;
  int tanPsi() const;

  int phiCMSSW() const;
  int phiBendCMSSW() const;

  int pathWireId(int) const;
  int pathTDC(int) const;
  int pathLat(int) const;

private:
  int m_xLocal;
  int m_tanPsi;

  int m_phiCMSSW;
  int m_phiBendCMSSW;

  int m_pathWireId[8];
  int m_pathTDC[8];
  int m_pathLat[8];
};

#endif

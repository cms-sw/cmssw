//-------------------------------------------------
//
//   Class L1MuDTChambThDigi
//
//   Description: input data for PHTF trigger
//
//
//   Author List: Jorge Troconiz  UAM Madrid
//
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"

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
L1MuDTChambThDigi::L1MuDTChambThDigi() {

  bx              = -100;
  wheel           = 0;
  sector          = 0;
  station         = 0;

  for(int i=0;i<7;i++) {
    m_outPos[i] = 0;
    m_outQual[i] = 0;
  }
}

L1MuDTChambThDigi::L1MuDTChambThDigi( int ubx, int uwh, int usc, int ust,
                                      int* upos, int* uqual ) {

  bx              = ubx;
  wheel           = uwh;
  sector          = usc;
  station         = ust;

  for(int i=0;i<7;i++) {
    m_outPos[i] = upos[i];
    m_outQual[i] = uqual[i];
  }
}

L1MuDTChambThDigi::L1MuDTChambThDigi( int ubx, int uwh, int usc, int ust,
                                      int* upos ) {

  bx              = ubx;
  wheel           = uwh;
  sector          = usc;
  station         = ust;

  for(int i=0;i<7;i++) {
    m_outPos[i] = upos[i];
    m_outQual[i] = 0;
  }
}

//--------------
// Destructor --
//--------------
L1MuDTChambThDigi::~L1MuDTChambThDigi() {
}

//--------------
// Operations --
//--------------
int L1MuDTChambThDigi::bxNum() const {
  return bx;
}

int L1MuDTChambThDigi::whNum() const {
  return wheel;
}
int L1MuDTChambThDigi::scNum() const {
  return sector;
}
int L1MuDTChambThDigi::stNum() const {
  return station;
}

int L1MuDTChambThDigi::code(const int i) const {
  if (i<0||i>=7) return 0;

  return (int)(m_outPos[i]+m_outQual[i]);
}

int L1MuDTChambThDigi::position(const int i) const {
  if (i<0||i>=7) return 0;

  return (int)m_outPos[i];
}

int L1MuDTChambThDigi::quality(const int i) const {
  if (i<0||i>=7) return 0;

  return (int)m_outQual[i];
}

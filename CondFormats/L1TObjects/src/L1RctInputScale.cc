// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1RctInputScale
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      
// Created:     Wed Sep 27 17:18:27 CEST 2006
// $Id: 

#include "CondFormats/L1TObjects/interface/L1RctInputScale.h"

using std::vector;
using std::ostream;
using std::endl;

/// construct a linear scale with a particular LSB
L1RctInputScale::L1RctInputScale(double lsb)
{
  for (unsigned i=0; i<nBinRank; i++) {
    for (unsigned eta=0; eta<nBinEta; eta++) {
      m_scale[i][eta] = lsb * i;
    }
  }
}

/// dtor
L1RctInputScale::~L1RctInputScale() {

}

/// set scale bin
void L1RctInputScale::setBin(unsigned short rank, unsigned short eta, double et) {
  if (rank < nBinRank && eta < nBinEta) {
    m_scale[rank][eta] = et;
  }
  else {
    // throw
  }
}

/// convert from Et in GeV to rank
unsigned short L1RctInputScale::rank(const double et, const unsigned short eta) const {
  if (eta < nBinEta) {
    unsigned short out = 0;
    for (unsigned i=0; i<nBinRank; i++) {
      if ( et >= m_scale[i][eta] ) { out = i; }
    }
    return out & nBinRank;
  }
  else {
    // throw
  }

}

// convert from rank to Et/GeV
double L1RctInputScale::et(const unsigned short rank, const unsigned short eta) const {
    if (rank < nBinRank && eta < nBinEta) {
      return m_scale[rank][eta];
  }
  else return -1.;
}

// pretty print
void L1RctInputScale::print(ostream& s) const {
  s << "L1RctInputScale" << endl;
  for (unsigned rank=0; rank<nBinRank; rank++) {
    s << rank << " ";
    for (unsigned eta=0; eta<nBinEta; eta++) {
      s << m_scale[rank][eta] << " ";
    }
    s << endl;
  }
}



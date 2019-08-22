// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1CaloHcalScale
//
// Implementation:
//     <Notes on implementation>
//
// Author:
// Created:     Wed Sep 27 17:18:27 CEST 2006
// $Id:

#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"

using std::endl;
using std::ostream;
using std::vector;

/// construct a linear scale with a particular LSB
L1CaloHcalScale::L1CaloHcalScale(double lsb) {
  for (unsigned i = 0; i < nBinRank; i++) {
    for (unsigned eta = 0; eta < nBinEta; eta++) {
      m_scale[i][eta] = lsb * i;
      m_scale[i][eta + nBinEta] = lsb * i;
    }
  }
}

/// dtor
L1CaloHcalScale::~L1CaloHcalScale() {}

/// set scale bin
void L1CaloHcalScale::setBin(unsigned short rank, unsigned short eta, short etaSign, double et) {
  --eta;  // input eta index starts at 1
  if (rank < nBinRank && eta < nBinEta) {
    if (etaSign < 0)
      eta += nBinEta;
    m_scale[rank][eta] = et;
  } else {
    // throw
  }
}

/// convert from Et in GeV to rank
unsigned short L1CaloHcalScale::rank(double et, unsigned short eta, short etaSign) const {
  --eta;  // input eta index starts at 1
  if (eta < nBinEta) {
    unsigned short out = 0;
    if (etaSign < 0)
      eta += nBinEta;
    for (unsigned i = 0; i < nBinRank; i++) {
      if (et >= m_scale[i][eta]) {
        out = i;
      }
    }
    return out & (nBinRank - 1);
  } else {
    // throw
  }
  return nBinRank;
}

// convert from rank to Et/GeV
double L1CaloHcalScale::et(unsigned short rank, unsigned short eta, short etaSign) const {
  --eta;  // input eta index starts at 1
  if (rank < nBinRank && eta < nBinEta) {
    if (etaSign < 0)
      eta += nBinEta;
    return m_scale[rank][eta];
  } else
    return -1.;
}

// pretty print
void L1CaloHcalScale::print(ostream& s) const {
  s << "L1CaloHcalScaleRcd" << endl;
  s << "Energy for HCAL inputs into the RCT" << endl;
  s << "Each new row is for a given value of 8 bit output of HCAL.  Each column is for the respective eta value "
    << endl;
  for (unsigned rank = 0; rank < nBinRank; rank++) {
    s << "rank " << rank << " ";
    for (unsigned eta = 0; eta < 2 * nBinEta; eta++) {
      s << m_scale[rank][eta] << " ";
    }
    s << endl;
  }
}

// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1CaloEtScale
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      
// Created:     Wed Sep 27 17:18:27 CEST 2006
// $Id: 

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

using std::vector;
using std::ostream;
using std::endl;

// default constructor (testing only!)
L1CaloEtScale::L1CaloEtScale() :
  m_linScaleMax(0x3ff),
  m_rankScaleMax(0x3f),
  m_linearLsb(1.0)
{
  for (unsigned i=0; i<m_rankScaleMax; i++) {
    m_thresholds.push_back(m_linearLsb * i);
  }
}

// ctor that provides backwards compatibility with fixed max scale values
// OK to use this with e/gamma and rank scales
L1CaloEtScale::L1CaloEtScale(const double linearLsbInGeV, const vector<double> thresholdsInGeV) :
  m_linScaleMax(0x3ff),
  m_rankScaleMax(0x3f),
  m_linearLsb(linearLsbInGeV),
  m_thresholds(thresholdsInGeV) {

  // protect against too many thresholds!
  //  while ( m_threshold.size() > (L1GctJetScale::maxRank+1) ) {
  //    m_thresholds.pop_back();
  //  }

}


// ctor that sets scale max values
L1CaloEtScale::L1CaloEtScale(const unsigned linScaleMax, const unsigned rankScaleMax, const double linearLsbInGeV, const vector<double> thresholdsInGeV) :
  m_linScaleMax(0x3ff),
  m_rankScaleMax(0x3f),
  m_linearLsb(linearLsbInGeV),
  m_thresholds(thresholdsInGeV) {

}


L1CaloEtScale::~L1CaloEtScale() {

}

// convert from linear Et to rank
uint16_t L1CaloEtScale::rank(const uint16_t linear) const {

  return rank( (linear & m_linScaleMax) * m_linearLsb);

}

/// convert from Et in GeV to rank
uint16_t L1CaloEtScale::rank(const double EtInGeV) const {

  uint16_t out = 0;

  for (unsigned i=0; i<m_thresholds.size() && i<(unsigned)(m_rankScaleMax+1); i++) {
    if ( EtInGeV >= m_thresholds[i] ) { out = i; }
  }

  return out & m_rankScaleMax;
}

// convert from rank to Et/GeV
double L1CaloEtScale::et(const uint16_t rank) const {

  // return bin centre, except for highest bin
//   if (rank < m_thresholds.size()-1) {
//     return (m_thresholds[rank+1]+m_thresholds[rank]) / 2;
//   }
//   else {
//     return m_thresholds.back();
//   }

// return bin lower edge
  return m_thresholds[rank];

}

void L1CaloEtScale::print(ostream& s) const {
  s << "L1CaloEtScale :" << endl;
  s << "  Input scale max = " << m_linScaleMax << endl;
  s << "  Input LSB       = " << m_linearLsb << " GeV" << endl;
  s << "  Rank scale max  = " << m_rankScaleMax << endl;
  for (unsigned i=0; i<m_thresholds.size(); i++) {
    s << "  Threshold " << i << " = " << m_thresholds[i] << " GeV" << endl;
  }
}



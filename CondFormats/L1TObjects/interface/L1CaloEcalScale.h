#ifndef L1TObjects_L1CaloEcalScale_h
#define L1TObjects_L1CaloEcalScale_h
// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1CaloEcalScale
//
/**\class L1CaloEcalScale L1CaloEcalScale.h CondFormats/L1TObjects/interface/L1CaloEcalScale.h

 Description: Class to handle conversion between Et scales in L1 hardware

 Usage:
    <usage>

*/
//
// Author:      Jim Brooke
// Created:     Wed Sep 27 17:18:27 CEST 2006
// $Id:
//

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <ostream>
#include <cstdint>

class L1CaloEcalScale {
public:
  //  static const unsigned short nBinRank = 0xff;
  static const unsigned short nBinRank = 1 << 8;
  static const unsigned short nBinEta = 28;  // per half, eta index is 1-28

  /// constructor (creates a linear scale with an LSB - no LSB gives identity)
  L1CaloEcalScale(double lsb = 1.0);

  /// destructor
  ~L1CaloEcalScale();

  // eta = |eta|
  // etaSign = +1 or -1

  /// set scale element; use this to create non-linear scales
  void setBin(unsigned short rank,
              unsigned short eta,  // input eta index is 1-28
              short etaSign,
              double et);

  /// convert from physical Et in GeV to rank scale
  uint16_t rank(double et,
                unsigned short eta,  // input eta index is 1-28
                short etaSign) const;

  /// convert from rank to physically meaningful quantity
  double et(unsigned short rank,
            unsigned short eta,  // input eta index is 1-28
            short etaSign) const;

  void print(std::ostream& s) const;

private:
  /// thresholds associated with rank scale in GeV
  // First nBinEta eta bins for positive eta, second nBinEta bins for negative
  double m_scale[nBinRank][2 * nBinEta];

  COND_SERIALIZABLE;
};

#endif

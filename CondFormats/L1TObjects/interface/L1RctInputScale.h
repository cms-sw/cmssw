#ifndef L1TObjects_L1RctInputScale_h
#define L1TObjects_L1RctInputScale_h
// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1RctInputScale
// 
/**\class L1RctInputScale L1RctInputScale.h CondFormats/L1TObjects/interface/L1RctInputScale.h

 Description: Class to handle conversion between Et scales in L1 hardware

 Usage:
    <usage>

*/
//
// Author:      Jim Brooke
// Created:     Wed Sep 27 17:18:27 CEST 2006
// $Id: 
//

#include <vector>
#include <ostream>

class L1RctInputScale {

 public:

  /// constructor (creates a linear scale with an LSB - no LSB gives identity)
  L1RctInputScale(double lsb=1.0);

  /// destructor
  ~L1RctInputScale();

  /// set scale element; use this to create non-linear scales
  void setBin(unsigned short rank, unsigned short eta, double et);

  /// convert from physical Et in GeV to rank scale
  uint16_t rank(const double et, const unsigned short eta) const;

  /// convert from rank to physically meaningful quantity
  double et(const unsigned short rank, const unsigned short eta) const;

  void print(std::ostream& s) const;

 private:

  static const unsigned short nBinRank = 0x3ff;
  static const unsigned short nBinEta = 11;

  /// thresholds associated with rank scale in GeV
  double m_scale[nBinRank][nBinEta];

};

#endif

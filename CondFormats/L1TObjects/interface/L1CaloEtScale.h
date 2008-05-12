#ifndef L1TObjects_L1CaloEtScale_h
#define L1TObjects_L1CaloEtScale_h
// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1CaloEtScale
// 
/**\class L1CaloEtScale L1CaloEtScale.h CondFormats/L1TObjects/interface/L1CaloEtScale.h

 Description: Class to handle conversion between Et scales in L1 hardware

 Usage:
    <usage>

*/
//
// Author:      Jim Brooke
// Created:     Wed Sep 27 17:18:27 CEST 2006
// $Id: 
//

#include <boost/cstdint.hpp>
#include <vector>
#include <ostream>

class L1CaloEtScale {

 public:

  /// default constructor (out = in)
  L1CaloEtScale();

  /// constructor takes physically meaningful quantities
  L1CaloEtScale(const double linearLsbInGeV, const std::vector<double> thresholdsInGeV);

  /// constructor for non-default number of bits
  L1CaloEtScale(unsigned linScaleMax, unsigned rankScaleMax, const double linearLsbInGeV, const std::vector<double> thresholdsInGeV);

  // destructor
  ~L1CaloEtScale();

  /// get LSB of linear input scale
  double linearLsb() const { return m_linearLsb; }

  /// convert from linear Et scale to rank scale
  uint16_t rank(const uint16_t linear) const;

  /// convert from physical Et in GeV to rank scale
  uint16_t rank(const double EtInGeV) const;

  /// convert from rank to physically meaningful quantity
  double et(const uint16_t rank) const;

  void print(std::ostream& s) const;

 private:

  /// linear scale maximum
  uint16_t m_linScaleMax;
  
  /// rank scale maximum
  uint16_t m_rankScaleMax;

  /// LSB of linear scale in GeV
  double m_linearLsb;

  /// thresholds associated with rank scale in GeV
  std::vector<double> m_thresholds;

};

#endif

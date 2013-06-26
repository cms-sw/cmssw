#ifndef L1TObjects_L1CaloEtScale_h
#define L1TObjects_L1CaloEtScale_h
// -*- C++ -*-
//
// Package:     L1TObjects
// Class  :     L1CaloEtScale
// 
/**\class L1CaloEtScale L1CaloEtScale.h CondFormats/L1TObjects/interface/L1CaloEtScale.h

 Description: Class to handle non-linear scales in L1 calo trigger hardware, including, e/gamma rank, jet rank, Htmiss


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

  /// default constructor, for testing (out = in)
  L1CaloEtScale();

  /// ctor that provides backwards compatibility with fixed scale max values
  /// OK to use this with e/gamma and jet rank scales
  L1CaloEtScale(const double linearLsbInGeV, const std::vector<double>& thresholdsInGeV);

  /// general case ctor that sets scale max values
  L1CaloEtScale(const unsigned linScaleMax, const unsigned rankScaleMax, const double linearLsbInGeV, const std::vector<double>& thresholdsInGeV);

  // destructor
  ~L1CaloEtScale();

  // get input scale size
  unsigned linScaleMax() const { return m_linScaleMax; }

  // get output scale size
  unsigned rankScaleMax() const { return m_rankScaleMax; }

  /// get LSB of linear input scale
  double linearLsb() const { return m_linearLsb; }

  /// convert from linear Et scale to rank scale
  uint16_t rank(const uint16_t linear) const;

  /// convert from physical Et in GeV to rank scale
  uint16_t rank(const double EtInGeV) const;

  /// convert from rank to physically meaningful quantity
  double et(const uint16_t rank) const;

  /// get thresholds
  const std::vector<double>& getThresholds() const { return m_thresholds; }

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

std::ostream& operator << (std::ostream& os, const L1CaloEtScale onj);

#endif

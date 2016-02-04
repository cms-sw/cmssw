#ifndef ECALEndcapProperties_H
#define ECALEndcapProperties_H

#include "FastSimulation/CalorimeterProperties/interface/ECALProperties.h"

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Pb, W and O
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

namespace edm { 
  class ParameterSet;
}

class ECALEndcapProperties : public ECALProperties 
{

 public:

  ECALEndcapProperties(const edm::ParameterSet& fastDet) ;

  virtual ~ECALEndcapProperties() { }

  /// Thickness (cm)
  double thickness(double eta) const { return 22.0; }

  ///Photostatistics (photons/GeV) in the homegeneous material
  inline double photoStatistics() const { return 50E3; }

  ///Light Collection efficiency [Default : 2.3%]
  inline double lightCollectionEfficiency() const { return lightColl; }

  ///Light Collection uniformity
  inline double lightCollectionUniformity() const {return 0.003;}

};

#endif

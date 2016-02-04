#ifndef ECALBarrelProperties_H
#define ECALBarrelProperties_H

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

class ECALBarrelProperties : public ECALProperties 
{

 public:

  ECALBarrelProperties(const edm::ParameterSet& fastDet);

  virtual ~ECALBarrelProperties() { }

  /// Thickness (in cm)
  double thickness(double eta) const { return 23.0; }

  ///Photostatistics (photons/GeV) in the homegeneous material
  inline double photoStatistics() const { return 50E3; }

  ///Light Collection efficiency [Default : 3.0%]
  inline double lightCollectionEfficiency() const { return lightColl; }

  ///Light Collection uniformity
  inline double lightCollectionUniformity() const {return 0.003;}

};

#endif

#ifndef HCALEndcapProperties_H
#define HCALEndcapProperties_H

#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Pb, W and O
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004  
*/ 

#include <cmath>

class HCALEndcapProperties : public HCALProperties 
{

 public:

  HCALEndcapProperties(const edm::ParameterSet& fastDet):HCALProperties(fastDet) {; } 

  virtual ~HCALEndcapProperties() { }

  double getHcalDepth(double);

  double thickness(const double eta) const { 
    return HCALProperties::getHcalDepth(eta) * interactionLength();
  }

 private:

};

#endif

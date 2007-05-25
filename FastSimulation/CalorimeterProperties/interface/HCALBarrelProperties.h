#ifndef HCALBarrelProperties_H
#define HCALBarrelProperties_H

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

namespace edm { 
  class ParameterSet;
}

class HCALBarrelProperties : public HCALProperties 
{

 public:

  HCALBarrelProperties(const edm::ParameterSet& fastDet):HCALProperties(fastDet) {; } 

  virtual ~HCALBarrelProperties() { }

  /// Thickness (in cm), according to a document of 1995. TO be checked.
  double thickness(double eta) const { 
    double e  = std::exp(-eta);
    double e2 = e*e;
    //    double c  = (1.-e2)/(1.+e2);
    double s  = 2.*e/(1.+e2);
    double t  = fabs(2.*e/(1.-e2));
    double feta = fabs(eta);

    if ( feta < 0.380 ) 
      {
      // This where the HO is supposed to be (1995 version)
	return 7.76 * interactionLength() / s ;  
      } 
    else if ( feta < 1.310 ) 
      {
	return 6.76 * interactionLength() / s ;  
      } 
    else if ( feta < 1.370 ) 
      {
	double e1  = std::exp(-1.310);
	double t1  = 2.*e1 /(1.-e1*e1);
	// 193.0 cm is the inner radius of HCAL
	return ( (6.76 * interactionLength() + 193.0) * t/t1 - 193.0) / s;
    } 
    //    else if ( feta < 1.440 ) 
    else if (feta < 1.450)  // F.B 12/01/05 : avoid edge effet 
                            // in ParticlePropagator, the limit is 1.44826
      {
	double e1  = std::exp(-1.310);
	double t1  = 2.*e1 /(1.-e1*e1);
	// 193.0 cm is the inner radius of HCAL
	return ( (6.76 * interactionLength() + 193.0) * t/t1 - 193.0) / (2.*s);
      } 
    else 
      {
	return 0.;
      }

  }

};

#endif

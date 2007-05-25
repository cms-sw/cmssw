#ifndef HCALForwardProperties_H
#define HCALForwardProperties_H

#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"

#include <cmath>

namespace edm { 
  class ParameterSet;
}

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Pb, W and O
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

class HCALForwardProperties : public HCALProperties
{

 public:

  HCALForwardProperties(const edm::ParameterSet& fastDet):HCALProperties(fastDet) {; } 

  virtual ~HCALForwardProperties() { }

  /// Radiation length in cm
  inline double radLenIncm()  const { return radiationLengthIncm(); }

  /// Radiation length in cm but static 
  static inline double radiationLengthIncm() { return 1.43; }
 
  /// Radiation length in g/cm^2
  inline double radLenIngcm2() const { return 12.86; }

  ///Interaction length in cm
  inline double interactionLength() const { return 15.05; }

  double thickness(double eta) const 
    { 
      double e  = std::exp(-eta);
      double e2 = e*e;
      // 1 / cos theta
      double cinv  = (1.+e2)/(1.-e2);
      //    double c  = (1.-e2)/(1.+e2);
      //    double s  = 2.*e/(1.+e2);
      //    double t  = 2.*e/(1.-e2);
       double feta = fabs(eta);
       if ( 3 < feta && feta < 5 ) 
      {
	return 165. * fabs(cinv); 
      }
    else
      {
	return 0;
      }
    }

 private:

};

#endif

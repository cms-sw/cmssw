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

class HCALEndcapProperties : public HCALProperties 
{

 public:

  HCALEndcapProperties(const edm::ParameterSet& fastDet):HCALProperties(fastDet) {; } 

  virtual ~HCALEndcapProperties() { }

  /// Thickness (in cm), according to a document of 1995. TO be checked.
  double thickness(const double eta) const { 

    double e  = exp(-eta);
    double e2 = e*e;
    double c  = (1.-e2)/(1.+e2);
    //    double s  = 2.*e/(1.+e2);
    double t  = fabs(2.*e/(1.-e2));
    double feta = fabs(eta);

    if ( 1.440 < feta && feta < 1.550 ) 
      {
	double e1  = exp(-1.550);
	double t1  = 2.*e1 /(1.-e1*e1);
	// 388.0 cm is the inner z of HCAL
	return ((10.78*interactionLength()+388.0) * t1/t - 388.0) / fabs(c);
      } 
    else if ( 1.550 <= feta && feta < 3. ) 
      {
	return 10.78 * interactionLength() / fabs(c) ;  
      } 
    else
      {
	return 0.;
      }
  }

 private:

};

#endif

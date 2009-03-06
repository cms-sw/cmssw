#ifndef HCALProperties_H
#define HCALProperties_H

#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Cu and Zn (brass) - For now
 * assume it is all Copper, and it'll be good enough.
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

namespace edm { 
  class ParameterSet;
}

class HCALProperties : public CalorimeterProperties 
{

 public:

  HCALProperties(const edm::ParameterSet& fastDet);

  virtual ~HCALProperties() {
  }

  /// Effective A
  inline double theAeff() const { return 63.546; }

  /// Effective Z
  inline double theZeff() const { return 29.; }

  /// Density in g/cm3
  inline double rho() const { return 8.960; }

  /// Radiation length in cm
  inline double radLenIncm()  const { return radiationLengthIncm(); }

  /// Radiation length in cm but static 
  // This is needed in Calorimetry/CrystalSegment. 
  // Patrick, if you don't like it, give me another solution
  // to access the ECALProperties efficiently. 
  static inline double radiationLengthIncm() { return 1.43; }
 
  /// Radiation length in g/cm^2
  inline double radLenIngcm2() const { return 12.86; }

  /// Moliere Radius in cm (=7 A/Z in g/cm^2)
  inline   double moliereRadius() const { return 1.712; }
  //inline double moliereRadius()  const { return 2.4; }

  /// Critical energy in GeV (2.66E-3*(x0*Z/A)^1.1)
  inline double criticalEnergy() const { return 18.63E-3; }

  ///Interaction length in cm
  inline double interactionLength() const { return 15.05; }

  ///h/pi Warning ! This is a ad-hoc parameter. It has been tuned to get a good agreement on 1TeV electrons
  ///It might have nothing to do with reality
  inline double hOverPi() const {return hOPi;}

  /// Spot fraction wrt ECAL 
  inline double spotFraction() const {return spotFrac;}

  double getHcalDepth(double) const;  

  int eta2ieta(double eta) const; 

 private:
  double hOPi;
  double spotFrac;


  double etatow[42];      // HCAL towers eta edges
  double hcalDepthLam[41]; // HCAL depth for each tower ieta
};

#endif

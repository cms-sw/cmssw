#ifndef HCALProperties_H
#define HCALProperties_H

#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"
#include <vector>

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
  inline double theAeff() const { return HCALAeff_; }

  /// Effective Z
  inline double theZeff() const { return HCALZeff_; }

  /// Density in g/cm3
  inline double rho() const { return HCALrho_; }

  /// Radiation length in cm
  inline double radLenIncm()  const { return radiationLengthIncm(); }

  /// Radiation length in cm but static 
  // This is needed in Calorimetry/CrystalSegment. 
  // Patrick, if you don't like it, give me another solution
  // to access the ECALProperties efficiently. 
  inline double radiationLengthIncm() const { return HCALradiationLengthIncm_; }
 
  /// Radiation length in g/cm^2
  inline double radLenIngcm2() const { return HCALradLenIngcm2_; }

  /// Moliere Radius in cm (=7 A/Z in g/cm^2)
  inline   double moliereRadius() const { return HCALmoliereRadius_; }
  //inline double moliereRadius()  const { return 2.4; }

  /// Critical energy in GeV (2.66E-3*(x0*Z/A)^1.1)
  inline double criticalEnergy() const { return HCALcriticalEnergy_; }

  ///Interaction length in cm
  inline double interactionLength() const { return HCALinteractionLength_; }

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

 protected:
  double HCALAeff_;
  double HCALZeff_;
  double HCALrho_;
  double HCALradiationLengthIncm_;
  double HCALradLenIngcm2_;
  double HCALmoliereRadius_;
  double HCALcriticalEnergy_;
  double HCALinteractionLength_;
  std::vector <double> etatow_;// HCAL towers eta edges
  std::vector <double> hcalDepthLam_;// HCAL depth for each tower ieta

};

#endif

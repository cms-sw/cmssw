#ifndef ECALProperties_H
#define ECALProperties_H

#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Pb, W and O
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

class ECALProperties : public CalorimeterProperties 
{

 public:

  ECALProperties() { } 

  virtual ~ECALProperties() {
  }

  /// Effective A
  inline double theAeff() const { return 170.87; }

  /// Effective Z
  inline double theZeff() const { return 68.36; }

  /// Density in g/cm3
  inline double rho() const { return 8.280; }

  /// Radiation length in cm
  //  inline double radLenIncm()  const { return radiationLengthIncm(); }
  inline double radLenIncm()  const { return 0.89; }

  /// Radiation length in cm but static 
  // This is needed in Calorimetry/CrystalSegment. Patrick, if you don't like it, give
  // me an other solution to access the ECALProperties efficiently. 
  // static inline double radiationLengthIncm() { return 0.89; }

  /// Radiation length in g/cm^2
  inline double radLenIngcm2() const { return 7.37; }

  /// Moliere Radius in cm
  inline   double moliereRadius() const { return 2.190; }
  //inline double moliereRadius()  const { return 2.4; }

  /// Critical energy in GeV (2.66E-3*(x0*Z/A)^1.1)
  inline double criticalEnergy() const { return 8.74E-3; }

  ///Interaction length in cm
  inline double interactionLength() const { return 18.5; }

 ///Photostatistics (photons/GeV) in the homegeneous material
  inline virtual double photoStatistics() const=0;

  ///Light Collection efficiency 
  inline virtual double lightCollectionEfficiency() const=0;

  ///Light Collection uniformity
  inline virtual double lightCollectionUniformity() const=0;

 protected:

  double lightColl;

};

#endif

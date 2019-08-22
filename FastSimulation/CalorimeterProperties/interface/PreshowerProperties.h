#ifndef PreshowerProperties_H
#define PreshowerProperties_H

#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Pb, W and O
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */

class PreshowerProperties : public CalorimeterProperties {
public:
  PreshowerProperties() { ; }

  ~PreshowerProperties() override { ; }

  /// Effective A
  inline double theAeff() const override { return 207.2; }
  /// Effective Z
  inline double theZeff() const override { return 82.; }
  /// Density in g/cm3
  inline double rho() const override { return 11.350; }
  /// Radiation length in cm
  inline double radLenIncm() const override { return 0.56; }
  /// Radiation length in g/cm^2
  inline double radLenIngcm2() const override { return 6.370; }
  /// Moliere Radius in cm
  inline double moliereRadius() const override { return 1.53; }
  /// Electron critical energy in GeV
  inline double criticalEnergy() const override { return 7.79E-3; }
  /// Muon critical energy in GeV
  //inline double muonCriticalEnergy() const { return 141.; }

  ///Interaction length in cm
  inline double interactionLength() const override { return 17.1; }

  /// Fraction of energy collected on sensitive detectors
  virtual double sensitiveFraction() const = 0;

  /// Number of Mips/GeV on sensitive detectors
  virtual double mipsPerGeV() const = 0;

protected:
  double thick;
  double mips;

private:
};

#endif

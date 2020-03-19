#ifndef HDShowerParametrization_H
#define HDShowerParametrization_H

#include "FastSimulation/CalorimeterProperties/interface/ECALProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include "FastSimulation/ShowerDevelopment/interface/HSParameters.h"
/** 
 * Hadronic Shower parametrization utilities according to 
 * G. Grindhammer et al. in a way implemeted in CMSJET
 *
 * \author Salavat Abdullin
 * \date: 20.10.2004
 */

class HDShowerParametrization {
public:
  HDShowerParametrization(const ECALProperties* ecal, const HCALProperties* hcal, const HSParameters* hadronshower)
      : theECAL(ecal), theHCAL(hcal), theHSParameters(hadronshower) {}

  virtual ~HDShowerParametrization() {}

  const ECALProperties* ecalProperties() const { return theECAL; }

  const HCALProperties* hcalProperties() const { return theHCAL; }

  const HSParameters* hsParameters() const { return theHSParameters; }

  // to distinguish between low- and high-energy case
  void setCase(int choice) {
    if (choice < 1 || choice > 2)
      theCase = 2;
    else
      theCase = choice;
  }

  // Minimal energy for the parameters calculation ( e < emin)
  double emin() const { return 2.; }
  // First  range for the parameters calculation   ( emin < e < mid)
  double emid() const { return 10.; }
  // Second range for the parameters calculation   ( emid < e < emax)
  double emax() const { return 500.; }

  double e1() const { return 0.35; }
  double e2() const { return 0.09; }
  double alpe1() const {
    if (theCase == 1)
      return 1.08;
    else
      return 1.30;
  }
  double alpe2() const {
    if (theCase == 1)
      return 0.24;
    else
      return 0.255;
  }
  double bete1() const {
    if (theCase == 1)
      return 0.478;
    else
      return 0.289;
  }
  double bete2() const {
    if (theCase == 1)
      return 0.135;
    else
      return 0.010;
  }
  double alph1() const {
    if (theCase == 1)
      return 1.17;
    else
      return 0.38;
  }
  double alph2() const {
    if (theCase == 1)
      return 0.21;
    else
      return 0.23;
  }
  double beth1() const {
    if (theCase == 1)
      return 2.10;
    else
      return 0.83;
  }
  double beth2() const {
    if (theCase == 1)
      return 0.72;
    else
      return 0.049;
  }
  double part1() const {
    if (theCase == 1)
      return 0.751;
    else
      return 0.509;
  }
  double part2() const {
    if (theCase == 1)
      return 0.177;
    else
      return 0.021;
  }
  double r1() const { return 0.0124; }
  double r2() const { return 0.359; }
  double r3() const { return 0.0511; }

private:
  const ECALProperties* theECAL;
  const HCALProperties* theHCAL;
  const HSParameters* theHSParameters;

  int theCase;
};

#endif

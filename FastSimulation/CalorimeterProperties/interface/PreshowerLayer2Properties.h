#ifndef PreshowerLayer2Properties_H
#define PreshowerLayer2Properties_H

#include "FastSimulation/CalorimeterProperties/interface/PreshowerProperties.h"

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

class PreshowerLayer2Properties : public PreshowerProperties 
{

 public:

  PreshowerLayer2Properties(const edm::ParameterSet& fastDet);

  ~PreshowerLayer2Properties() {
      ;}

  /// Fraction of energy collected on sensitive detectors
  inline double sensitiveFraction() const { return 0.00515; }

  /// Number of Mips/GeV [Default : 59.5 Mips/GeV or 0.7*24 MeV/Mips]
  inline double mipsPerGeV() const { return mips; }

  /// Thickness in cm (pretend it's all lead)
  /// Default : 0.506 cm at normal incidence
  double thickness(const double eta) const ;

    /// properties of the material between ES and EE; there is about 12 cm between the two.
  inline double pseeIntLenIncm() const {return pseeInteractionLength_;}
    /// properties of the material between ES and EE 
  inline double pseeRadLenIncm() const {return pseeradLenIncm_;}

 private:
    double pseeInteractionLength_;
    double pseeradLenIncm_;
};

#endif

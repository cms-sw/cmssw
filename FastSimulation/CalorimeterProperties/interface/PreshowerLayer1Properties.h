#ifndef PreshowerLayer1Properties_H
#define PreshowerLayer1Properties_H

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

class PreshowerLayer1Properties : public PreshowerProperties {
public:
  PreshowerLayer1Properties(const edm::ParameterSet& fastDet);

  ~PreshowerLayer1Properties() override { ; }

  /// Fraction of energy collected on sensitive detectors
  inline double sensitiveFraction() const override { return 0.0036; }

  /// Number of Mips/GeV [Default : 41.7 Mips/GeV or 24 MeV/Mips]
  inline double mipsPerGeV() const override { return mips; }

  /// Thickness in cm (Pretend it is all lead)
  /// Default : 1.02 cm at normal incidence
  double thickness(double eta) const override;
};

#endif

#ifndef PolyFit2DParametrizedMagneticField_h
#define PolyFit2DParametrizedMagneticField_h

/** \class PolyFit2DParametrizedMagneticField
 *
 *  Magnetic Field engine wrapper for V. Maroussov's 2D parametrization
 *  of the MT data.
 *
 *  $Date: 2008/04/23 14:49:56 $
 *  $Revision: 1.2 $
 *  \author N. Amapane
 */

#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class BFit; }


class PolyFit2DParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor
  PolyFit2DParametrizedMagneticField(double bVal = 4.01242188708911);

  /// Constructor. Parameters taken from a PSet
  PolyFit2DParametrizedMagneticField(const edm::ParameterSet& parameters);

  /// Destructor
  virtual ~PolyFit2DParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  magfieldparam::BFit* theParam;
};
#endif


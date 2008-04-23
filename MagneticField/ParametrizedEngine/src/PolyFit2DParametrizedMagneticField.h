#ifndef PolyFit2DParametrizedMagneticField_h
#define PolyFit2DParametrizedMagneticField_h

/** \class PolyFit2DParametrizedMagneticField
 *
 *  Magnetic Field engine wrapper for V. Maroussov's 2D parametrization
 *  of the MT data.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane
 */

#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class BFit; }


class PolyFit2DParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor
  PolyFit2DParametrizedMagneticField(double bVal);

  /// Constructor. Parameters taken from a PSet
  PolyFit2DParametrizedMagneticField(const edm::ParameterSet& parameters);

  /// Destructor
  virtual ~PolyFit2DParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  magfieldparam::BFit* theParam;
};
#endif


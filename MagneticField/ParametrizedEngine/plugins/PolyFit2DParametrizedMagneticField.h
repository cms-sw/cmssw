#ifndef PolyFit2DParametrizedMagneticField_h
#define PolyFit2DParametrizedMagneticField_h

/** \class PolyFit2DParametrizedMagneticField
 *
 *  Magnetic Field engine wrapper for V. Maroussov's 2D parametrization
 *  of the MT data.
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author N. Amapane
 */

#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class BFit; }


class PolyFit2DParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor. Fitted bVal for the nominal currents are:
  /// 2.0216; 3.5162;  3.8114; 4.01242188708911
  PolyFit2DParametrizedMagneticField(double bVal = 3.8114);

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


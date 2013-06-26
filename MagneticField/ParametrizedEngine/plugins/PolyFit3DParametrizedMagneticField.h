#ifndef PolyFit3DParametrizedMagneticField_h
#define PolyFit3DParametrizedMagneticField_h

/** \class PolyFit3DParametrizedMagneticField
 *
 *  Magnetic Field engine wrapper for V. Maroussov's 3D parametrization
 *  of the MT data.
 *
 *  $Date: 2011/04/16 10:20:40 $
 *  $Revision: 1.1 $
 *  \author N. Amapane
 */

#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class BFit3D; }


class PolyFit3DParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor. Fitted bVal for the nominal currents are:
  /// 2.0216; 3.5162;  3.8114; 4.01242188708911
  PolyFit3DParametrizedMagneticField(double bVal = 3.8114);

  /// Constructor. Parameters taken from a PSet
  PolyFit3DParametrizedMagneticField(const edm::ParameterSet& parameters);

  /// Destructor
  virtual ~PolyFit3DParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  magfieldparam::BFit3D* theParam;
};
#endif


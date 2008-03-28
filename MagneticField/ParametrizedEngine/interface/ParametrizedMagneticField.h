#ifndef MagneticField_ParametrizedEngine_ParametrizedMagneticField_h
#define MagneticField_ParametrizedEngine_ParametrizedMagneticField_h

/** \class ParametrizedMagneticField
 *
 *  Magnetic Field based on the Veikko Karimaki's Parametrization
 *
 *  $Date: 2007/07/02 11:48:59 $
 *  $Revision: 1.1 $
 *  \author M. Chiorboli - Universit\`a and INFN Catania
 */

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class ParametrizedMagneticField : public MagneticField
{
 public:
  ParametrizedMagneticField(float a, float l) {a_ = a; l_ = l;}
  virtual ~ParametrizedMagneticField() {;}

  /// Field value at specified global point, in Tesla


  GlobalVector inTesla (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  float a_;
  float l_;
  bool trackerField(const GlobalPoint& gp, double a, double l, GlobalVector& bxyz) const ;
  void ffunkti(float u, float* ff) const;

};

#endif

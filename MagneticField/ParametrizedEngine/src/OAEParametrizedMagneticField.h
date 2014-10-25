#ifndef ParametrizedEngine_OAEParametrizedMagneticField_h
#define ParametrizedEngine_OAEParametrizedMagneticField_h

/** \class OAEParametrizedMagneticField
 *
 *  Magnetic Field engine wrapper for V. Karimaki's "off-axis expansion"
 *  of the TOSCA field version 1103l_071212 (2, 3, 3.4, 3.8, 4 T)
 *  valid in the region r<1.15 m and |z|<2.8 m 
 *  For details, cf TkBfield.h
 *   
 *
 *  \author N. Amapane - Torino
 */

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TkBfield.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class TkBfield; }

class OAEParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor, pass value for nominal field
  explicit OAEParametrizedMagneticField(float B);

  /// Constructor, pass string for nominal field [deprecated]
  explicit OAEParametrizedMagneticField(std::string T="3_8T");

  /// Constructor. Parameters taken from a PSet [deprecated]
  explicit OAEParametrizedMagneticField(const edm::ParameterSet& parameters);

  /// Destructor
  virtual ~OAEParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  magfieldparam::TkBfield  theParam;
};
#endif

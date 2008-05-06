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
 *  $Date: 2008/03/28 16:49:25 $
 *  $Revision: 1.1 $
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

namespace edm { class ParameterSet; }
namespace magfieldparam { class TkBfield; }

class OAEParametrizedMagneticField : public MagneticField {
 public:
  /// Constructor 
  OAEParametrizedMagneticField(std::string & T="3_8T");

  /// Constructor. Parameters taken from a PSet
  OAEParametrizedMagneticField(const edm::ParameterSet& parameters);

  /// Destructor
  virtual ~OAEParametrizedMagneticField();
  
  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const;

 private:
  magfieldparam::TkBfield* theParam;
};
#endif

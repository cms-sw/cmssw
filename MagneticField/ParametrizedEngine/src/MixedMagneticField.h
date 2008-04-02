#ifndef MagneticField_MixedMagneticField_h
#define MagneticField_MixedMagneticField_h

/** \class MixedMagneticField
 *
 *   Temporary solution for a patchwork of Magnetic Fields.
 *   The result of this producer may be an unphisical field map!!!
 *   Use at your own risk!!!!
 *
 *  $Date: 2008/03/29 11:54:58 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */


#include "MagneticField/Engine/interface/MagneticField.h"

class MixedMagneticField : public MagneticField {
 public:

  MixedMagneticField(const MagneticField* param,
		     const MagneticField* full,
		     double scale);

  virtual ~MixedMagneticField() {}

  GlobalVector inTesla (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const {return true;}

 private:
  GlobalVector theField;
  const MagneticField* theParam;
  const MagneticField* theFull;
  double theScale;
};

#endif

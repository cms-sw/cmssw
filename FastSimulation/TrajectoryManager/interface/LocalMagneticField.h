#ifndef FastSimualtion_TrajectoryManager_LocalMagneticField_h
#define FastSimualtion_TrajectoryManager_LocalMagneticField_h

/** \class LocalMagneticField
 *
 *  A MagneticField engine that returns a constant programmable field value.
 *
 *  $Date: 2007/04/19 14:20:25 $
 *  $Revision: 1.1 $
 *  \author Patrick Janot, copied from N. Amapane - CERN
 */


#include "MagneticField/Engine/interface/MagneticField.h"

class LocalMagneticField : public MagneticField {
 public:

  ///Construct passing the Z field component in Tesla
  LocalMagneticField(double value);

  virtual ~LocalMagneticField() {}

  GlobalVector inTesla (const GlobalPoint& gp) const;

 private:
  GlobalVector theField;
};

#endif

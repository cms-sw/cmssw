#ifndef FastSimualtion_TrajectoryManager_LocalMagneticField_h
#define FastSimualtion_TrajectoryManager_LocalMagneticField_h

/** \class LocalMagneticField
 *
 *  A MagneticField engine that returns a constant programmable field value.
 *
 *  $Date: 2006/05/31 13:43:26 $
 *  $Revision: 1.1 $
 *  \author Patrick Janot, copied from N. Amapane - CERN
 */


#include "MagneticField/Engine/interface/MagneticField.h"

class LocalMagneticField : public MagneticField {
 public:

  ///Construct passing the Z field component in Tesla
  LocalMagneticField(double value);

  ~LocalMagneticField() override {}

  GlobalVector inTesla (const GlobalPoint& gp) const override;

 private:
  GlobalVector theField;
};

#endif

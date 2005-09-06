#ifndef VolumeBasedMagneticField_H
#define VolumeBasedMagneticField_H

/** \class VolumeBasedMagneticField
 *  Magnetic field based on a dedicated geometry of magnetic volumes
 *
 *  $Date: 2005/07/13 10:48:39 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - INFN Torino
 */

/* #include "MagneticField/BaseMagneticField/interface/MagneticFieldEngine.h" */

namespace seal {
  class Context;
}

class MagGeometry;

/* class VolumeBasedMagneticField : public MagneticFieldEngine { */
class VolumeBasedMagneticField {
public:
  /// Constructor
  VolumeBasedMagneticField(seal::Context* ic=0);

  /// Destructor
  ~VolumeBasedMagneticField();

  /// Get the field
  void field(const double *, double Bfield[3]);

private:
  MagGeometry * theGeometry;
};

#endif // VolumeBasedMagneticField_H

#ifndef MagneticField_VolumeBasedMagneticField_h
#define MagneticField_VolumeBasedMagneticField_h

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"

class VolumeBasedMagneticField : public MagneticField {
 public:
  //  VolumeBasedMagneticField(const DDCompactView & cpv);
  VolumeBasedMagneticField( const edm::ParameterSet& config,
			    std::vector<MagBLayer *> theBLayers,
			    std::vector<MagESector *> theESectors,
			    std::vector<MagVolume6Faces*> theBVolumes,
			    std::vector<MagVolume6Faces*> theEVolumes);
  virtual ~VolumeBasedMagneticField();
  GlobalVector inTesla ( const GlobalPoint& g) const;

  const MagVolume * findVolume(const GlobalPoint & gp) const;


 private:
  MagGeometry* field;
};

#endif

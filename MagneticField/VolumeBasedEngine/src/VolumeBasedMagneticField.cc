#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Vector/interface/GlobalVector.h"

VolumeBasedMagneticField::VolumeBasedMagneticField(  std::vector<MagBLayer *> theBLayers,
			     std::vector<MagESector *> theESectors,
			     std::vector<MagVolume6Faces*> theBVolumes,
			     std::vector<MagVolume6Faces*> theEVolumes){

  edm::ParameterSet p;
  p.addParameter<double>("findVolumeTolerance", 0.0);
  p.addParameter<bool>("cacheLastVolume", true);
  p.addUntrackedParameter<bool>("timerOn", false);
  
  field = new MagGeometry(p,theBLayers,theESectors,theBVolumes,theEVolumes);

}

GlobalVector VolumeBasedMagneticField::inTesla ( const GlobalPoint& g) const {
  GlobalVector gv =  field->fieldInTesla(g);
  return gv;
}

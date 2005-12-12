#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Vector/interface/GlobalVector.h"

VolumeBasedMagneticField::VolumeBasedMagneticField( const edm::ParameterSet& config,
						    std::vector<MagBLayer *> theBLayers,
						    std::vector<MagESector *> theESectors,
						    std::vector<MagVolume6Faces*> theBVolumes,
						    std::vector<MagVolume6Faces*> theEVolumes){
  
  field = new MagGeometry(config,theBLayers,theESectors,theBVolumes,theEVolumes);

}

GlobalVector VolumeBasedMagneticField::inTesla ( const GlobalPoint& g) const {
  GlobalVector gv =  field->fieldInTesla(g);
  return gv;
}

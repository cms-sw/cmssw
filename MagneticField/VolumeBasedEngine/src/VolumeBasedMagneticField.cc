#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

VolumeBasedMagneticField::VolumeBasedMagneticField( const edm::ParameterSet& config,
						    std::vector<MagBLayer *> theBLayers,
						    std::vector<MagESector *> theESectors,
						    std::vector<MagVolume6Faces*> theBVolumes,
						    std::vector<MagVolume6Faces*> theEVolumes){
  
  field = new MagGeometry(config,theBLayers,theESectors,theBVolumes,theEVolumes);

}

VolumeBasedMagneticField::~VolumeBasedMagneticField(){
  delete field;
}




GlobalVector VolumeBasedMagneticField::inTesla ( const GlobalPoint& g) const {
  GlobalVector gv =  field->fieldInTesla(g);
  return gv;
}

const MagVolume * VolumeBasedMagneticField::findVolume(const GlobalPoint & gp) const
{
  return field->findVolume(gp);
}

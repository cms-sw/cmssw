#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

VolumeBasedMagneticField::VolumeBasedMagneticField( const edm::ParameterSet& config,
						    std::vector<MagBLayer *> theBLayers,
						    std::vector<MagESector *> theESectors,
						    std::vector<MagVolume6Faces*> theBVolumes,
						    std::vector<MagVolume6Faces*> theEVolumes, float rMax, float zMax, const MagneticField* param) : 
  field(new MagGeometry(config,theBLayers,theESectors,theBVolumes,theEVolumes)), 
  maxR(rMax),
  maxZ(zMax),
  paramField(param)
{}

VolumeBasedMagneticField::~VolumeBasedMagneticField(){
  delete field;
}

GlobalVector VolumeBasedMagneticField::inTesla (const GlobalPoint& gp) const {

  // If parametrization of the inner region is available, use it.
  if (paramField && paramField->isDefined(gp)) return paramField->inTeslaUnchecked(gp);

  // If point is outside magfield map, return 0 field (not an error)
  if (!isDefined(gp))  return GlobalVector();

  return field->fieldInTesla(gp);
}

GlobalVector VolumeBasedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const{
  //same as above, but do not check range
  if (paramField && paramField->isDefined(gp)) return paramField->inTeslaUnchecked(gp);
  return field->fieldInTesla(gp);
}


const MagVolume * VolumeBasedMagneticField::findVolume(const GlobalPoint & gp) const
{
  return field->findVolume(gp);
}


bool VolumeBasedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (fabs(gp.z()) < maxZ && gp.perp() < maxR);
}


bool VolumeBasedMagneticField::isZSymmetric() const {
  return field->isZSymmetric();
}

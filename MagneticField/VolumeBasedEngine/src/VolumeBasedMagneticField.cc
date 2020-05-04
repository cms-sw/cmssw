#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

VolumeBasedMagneticField::VolumeBasedMagneticField(int geomVersion,
                                                   const std::vector<MagBLayer*>& theBLayers,
                                                   const std::vector<MagESector*>& theESectors,
                                                   const std::vector<MagVolume6Faces*>& theBVolumes,
                                                   const std::vector<MagVolume6Faces*>& theEVolumes,
                                                   float rMax,
                                                   float zMax,
                                                   const MagneticField* param,
                                                   bool isParamFieldOwned)
    : field(new MagGeometry(geomVersion, theBLayers, theESectors, theBVolumes, theEVolumes)),
      maxRsq(rMax * rMax),
      maxZ(zMax),
      paramField(param),
      magGeomOwned(true),
      paramFieldOwned(isParamFieldOwned) {}

VolumeBasedMagneticField::VolumeBasedMagneticField(const VolumeBasedMagneticField& vbf)
    : MagneticField::MagneticField(vbf),
      field(vbf.field),
      maxRsq(vbf.maxRsq),
      maxZ(vbf.maxZ),
      paramField(vbf.paramField),
      magGeomOwned(false),
      paramFieldOwned(false) {
  // std::cout << "VolumeBasedMagneticField::clone() (shallow copy)" << std::endl;
}

MagneticField* VolumeBasedMagneticField::clone() const { return new VolumeBasedMagneticField(*this); }

VolumeBasedMagneticField::~VolumeBasedMagneticField() {
  if (magGeomOwned)
    delete field;
  if (paramFieldOwned)
    delete paramField;
}

GlobalVector VolumeBasedMagneticField::inTesla(const GlobalPoint& gp) const {
  // If parametrization of the inner region is available, use it.
  if (paramField && paramField->isDefined(gp))
    return paramField->inTeslaUnchecked(gp);

  // If point is outside magfield map, return 0 field (not an error)
  if (!isDefined(gp))
    return GlobalVector();

  return field->fieldInTesla(gp);
}

GlobalVector VolumeBasedMagneticField::inTesla(const GlobalPoint& gp, MagneticFieldCache& cache) const {
  // If parametrization of the inner region is available, use it.
  if (paramField && paramField->isDefined(gp))
    return paramField->inTeslaUnchecked(gp, cache);

  // If point is outside magfield map, return 0 field (not an error)
  if (!isDefined(gp))
    return GlobalVector();

  return field->fieldInTesla(gp, cache);
}

GlobalVector VolumeBasedMagneticField::inTeslaUnchecked(const GlobalPoint& gp) const {
  //same as above, but do not check range
  if (paramField && paramField->isDefined(gp))
    return paramField->inTeslaUnchecked(gp);
  return field->fieldInTesla(gp);
}

GlobalVector VolumeBasedMagneticField::inTeslaUnchecked(const GlobalPoint& gp, MagneticFieldCache& cache) const {
  //same as above, but do not check range
  if (paramField && paramField->isDefined(gp))
    return paramField->inTeslaUnchecked(gp, cache);
  return field->fieldInTesla(gp, cache);
}

const MagVolume* VolumeBasedMagneticField::findVolume(const GlobalPoint& gp) const { return field->findVolume(gp); }

bool VolumeBasedMagneticField::isDefined(const GlobalPoint& gp) const {
  return (fabs(gp.z()) < maxZ && gp.perp2() < maxRsq);
}

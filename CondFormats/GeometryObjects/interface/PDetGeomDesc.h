#ifndef CondFormats_PDetGeomDesc_h
#define CondFormats_PDetGeomDesc_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDesc.h"
#include "DetectorDescription/DDCMS/interface/DDTranslation.h"
#include "DetectorDescription/DDCMS/interface/DDRotationMatrix.h"

#include <vector>
#include <string>

class PDetGeomDesc {
public:
  struct Item {
    Item() = default;
    Item(const DetGeomDesc* const geoInfo) {
      dx_ = geoInfo->translation().X();
      dy_ = geoInfo->translation().Y();
      dz_ = geoInfo->translation().Z();

      const DDRotationMatrix& rot = geoInfo->rotation();
      rot.GetComponents(axx_, axy_, axz_, ayx_, ayy_, ayz_, azx_, azy_, azz_);
      name_ = geoInfo->name();
      params_ = geoInfo->params();
      copy_ = geoInfo->copyno();
      z_ = geoInfo->parentZPosition();
      sensorType_ = geoInfo->sensorType();
      geographicalID_ = geoInfo->geographicalID();
    }

    // Translation matrix elements
    double dx_, dy_, dz_;  // in mm
    // Rotation matrix elements
    double axx_, axy_, axz_, ayx_, ayy_, ayz_, azx_, azy_, azz_;
    std::string name_;
    std::vector<double> params_;  // default unit from DD4hep (cm)
    uint32_t geographicalID_;     // to be converted to DetId
    int copy_;
    float z_;  // in mm
    std::string sensorType_;

    COND_SERIALIZABLE;
  };

  std::vector<Item> container_;

  COND_SERIALIZABLE;
};

#endif

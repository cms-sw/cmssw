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
  PDetGeomDesc(){};
  ~PDetGeomDesc(){};

  struct Item {
    Item() {}
    Item(DetGeomDesc* const geoInfo) {
      dx_ = geoInfo->translation().X();
      dy_ = geoInfo->translation().Y();
      dz_ = geoInfo->translation().Z();

      const DDRotationMatrix& rot = geoInfo->rotation();
      double xx, xy, xz, yx, yy, yz, zx, zy, zz;
      rot.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz);
      axx_ = xx;
      axy_ = xy;
      axz_ = xz;
      ayx_ = yx;
      ayy_ = yy;
      ayz_ = yz;
      azx_ = zx;
      azy_ = zy;
      azz_ = zz;
      name_ = geoInfo->name();
      copy_ = geoInfo->copyno();
      z_ = geoInfo->parentZPosition();
      sensorType_ = geoInfo->sensorType();
      geographicalID_ = geoInfo->geographicalID();
    }

    // Translation matrix elements
    double dx_, dy_, dz_;
    // Rotation matrix elements
    double axx_, axy_, axz_, ayx_, ayy_, ayz_, azx_, azy_, azz_;
    std::string name_;
    std::vector<double> params_;
    uint32_t geographicalID_;  // to be converted to DetId
    int copy_;
    float z_;
    std::string sensorType_;

    COND_SERIALIZABLE;
  };

  std::vector<Item> container_;

  COND_SERIALIZABLE;
};

#endif

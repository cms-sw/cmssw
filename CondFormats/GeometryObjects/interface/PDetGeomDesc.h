#ifndef CondFormats_PDetGeomDesc_h
#define CondFormats_PDetGeomDesc_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <string>

class PDetGeomDesc {
public:
  PDetGeomDesc(){};
  ~PDetGeomDesc(){};

  struct Item {
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

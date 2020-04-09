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
    double _dx, _dy, _dz;
    // Rotation matrix elements
    double _axx, _axy, _axz, _ayx, _ayy, _ayz, _azx, _azy, _azz;
    std::string _name;  // save only the name, not the namespace.
    std::vector<double> _params;
    uint32_t _geographicalID;  // to be converted to DetId
    int _copy;
    float _z;
    std::string _sensorType;

    COND_SERIALIZABLE;
  };

  std::vector<Item> _container;

  COND_SERIALIZABLE;
};

#endif

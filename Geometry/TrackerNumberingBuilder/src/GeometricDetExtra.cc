#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"

/**
 * Constructors to be used when looping over DDD
 */
GeometricDetExtra::GeometricDetExtra(GeometricDet const* gd,
                                     DetId id,
                                     GeoHistory& gh,
                                     double vol,
                                     double dens,
                                     double wgt,
                                     double cpy,
                                     const std::string& mat,
                                     const std::string& name,
                                     bool dd)
    : _mygd(gd),
      _geographicalId(id),
      _parents(gh),
      _volume(vol),
      _density(dens),
      _weight(wgt),
      _copy((int)(cpy)),
      _material(mat),
      _name(name),
      _fromDD(dd) {}

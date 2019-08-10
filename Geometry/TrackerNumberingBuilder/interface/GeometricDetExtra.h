#ifndef Geometry_TrackerNumberingBuilder_GeometricDetExtra_H
#define Geometry_TrackerNumberingBuilder_GeometricDetExtra_H

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>
/**
 * Composite class GeometricDetExtra. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricDetExtra {
public:
#ifdef PoolAlloc
  using GeoHistory = std::vector<DDExpandedNode, PoolAlloc<DDExpandedNode> >;
#else
  using GeoHistory = std::vector<DDExpandedNode>;
#endif
  /**
   * Constructors to be used when looping over DDD
   */
  explicit GeometricDetExtra(GeometricDet const* gd,
                             DetId id,
                             GeoHistory& gh,
                             double vol,
                             double dens,
                             double wgt,
                             double cpy,
                             const std::string& mat,
                             const std::string& name,
                             bool dd = false);

  /**
   * get and set associated GeometricDet 
   * DOES NO CHECKING!
   */
  GeometricDet const* geometricDet() const { return _mygd; }

  /**
   * set or add or clear components
   */
  void setGeographicalId(DetId id) { _geographicalId = id; }
  DetId geographicalId() const { return _geographicalId; }
  GeoHistory const& parents() const { return _parents; }
  int copyno() const { return _copy; }
  double volume() const { return _volume; }
  double density() const { return _density; }
  double weight() const { return _weight; }
  std::string const& material() const { return _material; }

  /**
   * what it says... used the DD in memory model to build the geometry... or not.
   */
  bool wasBuiltFromDD() const { return _fromDD; }

  std::string const& name() const { return _name; }

private:
  /** Data members **/

  GeometricDet const* _mygd;
  DetId _geographicalId;
  GeoHistory _parents;
  double _volume;
  double _density;
  double _weight;
  int _copy;
  std::string _material;
  std::string _name;
  bool _fromDD;  // may not need this, keep an eye on it.
};

#undef PoolAlloc
#endif

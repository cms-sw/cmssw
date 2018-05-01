#ifndef Geometry_MTDNumberingBuilder_GeometricTimingDetExtra_H
#define Geometry_MTDNumberingBuilder_GeometricTimingDetExtra_H

#include "Geometry/MTDNumberingBuilder/interface/GeometricTimingDet.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>

class GeometricTimingDetExtra {
 public:
  typedef DDExpandedView::NavRange NavRange;
#ifdef PoolAlloc
  typedef std::vector< DDExpandedNode, PoolAlloc<DDExpandedNode> > GeoHistory;
#endif
#ifndef PoolAlloc
  typedef std::vector<DDExpandedNode> GeoHistory;
#endif
  /**
   * Constructors to be used when looping over DDD
   */
  GeometricTimingDetExtra( GeometricTimingDet const *gd ) : _mygd(gd) { }; // this better be "copied into" or it will never have any valid numbers/info.
    
    GeometricTimingDetExtra( GeometricTimingDet const *gd, DetId id, GeoHistory& gh,  double vol, double dens, double wgt, double cpy, const std::string& mat, const std::string& name, bool dd=false );

  /**
   *
   */
  ~GeometricTimingDetExtra();
  
  /**
   * get and set associated GeometricTimingDet 
   * DOES NO CHECKING!
   */
  GeometricTimingDet const * geometricDet() const { return _mygd; } 
  
  /**
   * set or add or clear components
   */
  void setGeographicalId(DetId id) {
    _geographicalId = id; 
  }
  DetId geographicalId() const { return _geographicalId; }

  GeoHistory const &  parents() const {

    return _parents;
  }
  //rr  
  int copyno() const {
    return _copy;
  }
  double volume() const {
    return _volume;
  }
  double density() const {
    return _density;
  }
  double weight() const {
    return _weight;
  }
  std::string const &  material() const {
    return _material;
  }
  
  /**
   * what it says... used the DD in memory model to build the geometry... or not.
   */
  bool wasBuiltFromDD() const {
    return _fromDD;
  }

  std::string const& name() const { return _name; }
  
 private:

  /** Data members **/

  GeometricTimingDet const* _mygd;  
  DetId _geographicalId;
  GeoHistory _parents;
  double _volume;
  double _density;
  double _weight;
  int    _copy;
  std::string _material;
  std::string _name;
  bool _fromDD; // may not need this, keep an eye on it.
};

#undef PoolAlloc
#endif

#ifndef Geometry_TrackerNumberingBuilder_GeometricDetExtra_H
#define Geometry_TrackerNumberingBuilder_GeometricDetExtra_H

//#include "CondFormats/GeometryObjects/interface/PGeometricDetExtra.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
/* #include "DetectorDescription/Base/interface/DDRotationMatrix.h" */
/* #include "DetectorDescription/Base/interface/DDTranslation.h" */
/* #include "DetectorDescription/Core/interface/DDSolidShapes.h" */
/* #include "DataFormats/GeometrySurface/interface/Surface.h" */
/* #include "DataFormats/GeometrySurface/interface/Bounds.h" */
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>
// waiting for template-alias
//#define PoolAlloc  __gnu_cxx::__pool_alloc
/**
 * Composite class GeometricDetExtra. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricDetExtra {
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
  GeometricDetExtra( GeometricDet const *gd ) : _mygd(gd) { }; // this better be "copied into" or it will never have any valid numbers/info.
    
    GeometricDetExtra( GeometricDet const *gd, DetId id, GeoHistory& gh,  double vol, double dens, double wgt, double cpy, const std::string& mat, const std::string& name, bool dd=false );

  /**
   *
   */
  ~GeometricDetExtra();
  
  /*
    GeometricDetExtra(const GeometricDetExtra &);
  
  GeometricDetExtra & operator=( const GeometricDetExtra & );
  */
  /**
   * get and set associated GeometricDet 
   * DOES NO CHECKING!
   */
  GeometricDet const * geometricDet() const { return _mygd; } 
  //  void setGeometricDet( GeometricDet* gd )  { _mygd=gd; }
  
  /**
   * set or add or clear components
   */
  void setGeographicalId(DetId id) {
    _geographicalId = id; 
    //std::cout <<"setGeographicalId " << int(id) << std::endl;
  }
  DetId geographicalId() const { return _geographicalId; }
  //rr
  /** parents() retuns the geometrical history
   * mec: only works if this is built from DD and not from reco DB.
   */
  GeoHistory const &  parents() const {
    //std::cout<<"parents"<<std::endl;
    return _parents;
  }
  //rr  
  int copyno() const {
    //std::cout<<"copyno"<<std::endl;
    return _copy;
  }
  double volume() const {
    //std::cout<<"volume"<<std::endl;
    return _volume;
  }
  double density() const {
    //std::cout<<"density"<<std::endl;
    return _density;
  }
  double weight() const {
    //std::cout<<"weight"<<std::endl;
    return _weight;
  }
  std::string const &  material() const {
    //std::cout<<"material"<<std::endl;
    return _material;
  }
  
  /**
   * what it says... used the DD in memory model to build the geometry... or not.
   */
  bool wasBuiltFromDD() const {
    //std::cout<<"wasBuildFromDD"<<std::endl;
    return _fromDD;
  }

  std::string const& name() const { return _name; }
  
 private:

  /** Data members **/

  GeometricDet const* _mygd;  
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

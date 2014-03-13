#ifndef Geometry_TrackerNumberingBuilder_GeometricDet_H
#define Geometry_TrackerNumberingBuilder_GeometricDet_H

#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <memory>
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>
// waiting for template-alias
//#define PoolAlloc  __gnu_cxx::__pool_alloc
// MEC: testing whether we need all bits-and-pieces.  Setting this makes GeometricDet like it used to be.
//#define GEOMETRICDETDEBUG
//class DetId;
class DDFilteredView;

/**
 * Composite class GeometricDet. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricDet {
 public:

  typedef DDExpandedView::nav_type DDnav_type;
  typedef DDExpandedView::NavRange NavRange;

  typedef std::vector< GeometricDet const *>  ConstGeometricDetContainer;
  typedef std::vector< GeometricDet *>  GeometricDetContainer;

#ifdef PoolAlloc  
  typedef std::vector< DDExpandedNode, PoolAlloc<DDExpandedNode> > GeoHistory;
  typedef std::vector<int, PoolAlloc<int> > nav_type;
#else
  typedef std::vector<DDExpandedNode> GeoHistory;
  typedef DDExpandedView::nav_type nav_type;
#endif
  typedef Surface::PositionType Position;
  typedef Surface::RotationType Rotation;

  //
  // more can be added; please add at the end!
  //
  typedef enum GDEnumType {unknown=100, Tracker=0, PixelBarrel=1, PixelEndCap=2,
			  TIB=3, TID=4, TOB=5, TEC=6,
			  layer=8, wheel=9, strng=10, rod=11, petal=12, ring=13,
			   ladder=14, mergedDet=15, DetUnit=16, disk=17, panel=18, PixelEndCapPhase1=20 } GeometricEnumType;
  /**
   * Constructors to be used when looping over DDD
   */
#ifdef GEOMETRICDETDEBUG
  GeometricDet(DDnav_type const & navtype, GeometricEnumType dd);
  GeometricDet(DDExpandedView* ev, GeometricEnumType dd);
#endif
  GeometricDet(DDFilteredView* fv, GeometricEnumType dd);
  GeometricDet(const PGeometricDet::Item& onePGD, GeometricEnumType dd);
  
  /*
  GeometricDet(const GeometricDet &);

  GeometricDet & operator=( const GeometricDet & );
  */

  /**
   * set or add or clear components
   */
  void setGeographicalID(DetId id) {
    _geographicalID = id; 
    //std::cout <<"setGeographicalID " << int(id) << std::endl;
  }
#ifdef GEOMETRICDETDEBUG
  void setComponents(GeometricDetContainer const & cont) {
    _container = cont; 
    //std::cout <<"setComponents" << std::endl;
  }
#endif
  void addComponents(GeometricDetContainer const & cont);
  void addComponents(ConstGeometricDetContainer const & cont);
  void addComponent(GeometricDet*);
  /**
   * clearComponents() only empties the container, the components are not deleted!
   */
  void clearComponents() {
    _container.clear();
    //std::cout<<"clearComponents"<<std::endl;
  }
 
  /**
   * deleteComponents() explicitly deletes the daughters
   * 
   */
  void deleteComponents();

  bool isLeaf() const { 
    //std::cout <<"isLeaf() ==" <<_container.empty()<<std::endl; 
    return _container.empty(); 
  }
  
  GeometricDet* component(size_t index) {
    return const_cast<GeometricDet*>(_container[index]);
  }

  /**
   * Access methods
   */
  DDRotationMatrix const & rotation() const {
    //std::cout<<"rotation" <<std::endl; 
    return _rot;
  }
  DDTranslation const & translation() const {
    //std::cout<<"translation" <<std::endl;
    return _trans;
  }
  double phi() const { 
    //std::cout<<"phi"<<std::endl; 
    return _phi; 
  }
  double rho() const { 
    //std::cout << "rho" <<std::endl; 
    return _rho; 
  }

  DDSolidShape const & shape() const  {
    //std::cout<<"shape"<<std::endl;
    return _shape;
  }
  GeometricEnumType type() const {
    //std::cout<<"type"<<std::endl;
    return _type;
  }
  DDName const & name() const {
    //std::cout<<"name"<<std::endl;
    return _ddname;
  }
  // internal representaion
  nav_type const & navType() const {
    //std::cout<<"navType"<<std::endl; 
    return _ddd;
  }
  // representation neutral interface
  NavRange navRange() const {
    //std::cout<<"navRange"<<std::endl;
    return NavRange(&_ddd.front(),_ddd.size());
  }
  // more meaningfull name (maybe)
  NavRange navpos() const {
    //std::cout<<"navpos"<<std::endl;
    return NavRange(&_ddd.front(),_ddd.size());
  }
  std::vector<double> const & params() const {
    //std::cout<<"params"<<std::endl;
    return _params;
  }

  ~GeometricDet();
  
  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricDet it returns nothing (an empty vector)
   */
  ConstGeometricDetContainer & components() {
    //std::cout << "components1" <<std::endl;
    return _container;
  }  
  ConstGeometricDetContainer const & components() const {
    //std::cout<<"const components2 "<<std::endl;
    return _container;
  }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricDet is a leaf, it returns it!
   */

  ConstGeometricDetContainer deepComponents() const;
  void deepComponents(ConstGeometricDetContainer & cont) const;

#ifdef GEOMETRICDETDEBUG
  //rr
  /** parents() retuns the geometrical history
   * mec: only works if this is built from DD and not from reco DB.
   */
  GeoHistory const &  parents() const {
    //std::cout<<"parents"<<std::endl;
    return _parents;
  }
  //rr  
#endif
  
  /**
   *geometricalID() returns the ID associated to the GeometricDet.
   */
  DetId geographicalID() const  { 
    //std::cout<<"geographicalID"<<std::endl;
    return _geographicalID; 
  }
  DetId geographicalId() const  { 
    //std::cout<<"geographicalId"<<std::endl; 
    return _geographicalID; 
  }

  /**
   *positionBounds() returns the position in cm. 
   */
  Position positionBounds() const; 

  /**
   *rotationBounds() returns the rotation matrix. 
   */
  Rotation  rotationBounds() const; 

  /**
   *bounds() returns the Bounds.
   */
  std::unique_ptr<Bounds> bounds() const; 
#ifdef GEOMETRICDETDEBUG
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
#endif
  double radLength() const {
    //std::cout<<"radLength"<<std::endl;
    return _radLength;
  }
  double xi() const {
    //std::cout<<"xi - chi"<<std::endl;
    return _xi;
  }
  /**
   * The following four pix* methods only return meaningful results for pixels.
   */
  double pixROCRows() const {
    //std::cout<<"pixROCRows"<<std::endl;
    return _pixROCRows;
  }
  double pixROCCols() const {
    //std::cout<<"pixROCCols"<<std::endl;
    return _pixROCCols;
  }
  double pixROCx() const {
    //std::cout<<"pixROCx"<<std::endl;
    return _pixROCx;
  }
  double pixROCy() const {
    //std::cout<<"pixROCy"<<std::endl;
    return _pixROCy;
  }

  /**
   * The following two are only meaningful for the silicon tracker.
   */  
  bool stereo() const {
    //std::cout<<"stereo"<<std::endl;
    return _stereo;
  }
  double siliconAPVNum() const {
    //std::cout<<"siliconAPVNum"<<std::endl;
    return _siliconAPVNum;
  }

  /**
   * what it says... used the DD in memory model to build the geometry... or not.
   */
#ifdef GEOMETRICDETDEBUG
  bool wasBuiltFromDD() const {
    //std::cout<<"wasBuildFromDD"<<std::endl;
    return _fromDD;
  }
#endif  

 private:

  ConstGeometricDetContainer _container;
  DDTranslation _trans;
  double _phi;
  double _rho;
  DDRotationMatrix _rot;
  DDSolidShape _shape;
  nav_type _ddd;
  DDName _ddname;
  GeometricEnumType _type;
  std::vector<double> _params;

  DetId _geographicalID;
#ifdef GEOMETRICDETDEBUG
  GeoHistory _parents;
  double _volume;
  double _density;
  double _weight;
  int    _copy;
  std::string _material;
#endif
  double _radLength;
  double _xi;
  double _pixROCRows;
  double _pixROCCols;
  double _pixROCx;
  double _pixROCy;
  bool _stereo;
  double _siliconAPVNum;
#ifdef GEOMETRICDETDEBUG
  bool _fromDD;
#endif

};

#undef PoolAlloc
#endif


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
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>

// waiting for template-alias
//#define PoolAlloc  __gnu_cxx::__pool_alloc
// MEC: testing whether we need all bits-and-pieces.  Setting this makes GeometricDet like it used to be.
//#define GEOMETRICDETDEBUG

class DDFilteredView;

/**
 * Composite class GeometricDet. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricDet
{
public:

  typedef DDExpandedView::nav_type DDnav_type;
  typedef DDExpandedView::NavRange NavRange;

  typedef std::vector< GeometricDet const *>  ConstGeometricDetContainer;
  typedef std::vector< GeometricDet const *>  GeometricDetContainer;

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
  
  /**
   * set or add or clear components
   */
  void setGeographicalID(DetId id) const {
    _geographicalID = id; 
  }

  void addComponents(GeometricDetContainer const & cont);
  void addComponent(GeometricDet*);

  /**
   * clearComponents() only empties the container, the components are not deleted!
   */
  void clearComponents() {
    _container.clear();
  }
 
  /**
   * deleteComponents() explicitly deletes the daughters
   * 
   */
  void deleteComponents();

  bool isLeaf() const { 
    return _container.empty(); 
  }
  
  DDRotationMatrix const & rotation() const {
    return _rot;
  }
  DDTranslation const & translation() const {
    return _trans;
  }
  double phi() const { 
    return _phi; 
  }
  double rho() const { 
    return _rho; 
  }

  DDSolidShape const & shape() const  {
    return _shape;
  }
  GeometricEnumType type() const {
    return _type;
  }
  DDName const & name() const {
    return _ddname;
  }
  // internal representaion
  nav_type const & navType() const {
    return _ddd;
  }
  // representation neutral interface
  NavRange navRange() const {
    return NavRange(&_ddd.front(),_ddd.size());
  }
  // more meaningfull name (maybe)
  NavRange navpos() const {
    return NavRange(&_ddd.front(),_ddd.size());
  }
  std::vector<double> const & params() const {
    return _params;
  }

  ~GeometricDet();
  
  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricDet it returns nothing (an empty vector)
   */
  GeometricDetContainer & components() {
    return _container;
  }  
  GeometricDetContainer const & components() const {
    return _container;
  }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricDet is a leaf, it returns it!
   */

  ConstGeometricDetContainer deepComponents() const;
  void deepComponents(GeometricDetContainer & cont) const;

  /**
   *geometricalID() returns the ID associated to the GeometricDet.
   */
  DetId geographicalID() const  { 
    return _geographicalID; 
  }
  DetId geographicalId() const  { 
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
  Bounds * const bounds() const; 

  double radLength() const {
    return _radLength;
  }
  double xi() const {
    return _xi;
  }
  /**
   * The following four pix* methods only return meaningful results for pixels.
   */
  double pixROCRows() const {
    return _pixROCRows;
  }
  double pixROCCols() const {
    return _pixROCCols;
  }
  double pixROCx() const {
    return _pixROCx;
  }
  double pixROCy() const {
    return _pixROCy;
  }

  /**
   * The following two are only meaningful for the silicon tracker.
   */  
  bool stereo() const {
    return _stereo;
  }
  double siliconAPVNum() const {
    return _siliconAPVNum;
  }

  /**
   * what it says... used the DD in memory model to build the geometry... or not.
   */

#ifdef GEOMETRICDETDEBUG
  void setComponents(GeometricDetContainer const & cont) {
    _container = cont; 
  }
  /** parents() retuns the geometrical history
   * mec: only works if this is built from DD and not from reco DB.
   */
  GeoHistory const &  parents() const {
    return _parents;
  }
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
  bool wasBuiltFromDD() const {
    return _fromDD;
  }
#endif  

private:

  GeometricDetContainer _container;
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
  double _radLength;
  double _xi;
  double _pixROCRows;
  double _pixROCCols;
  double _pixROCx;
  double _pixROCy;
  bool _stereo;
  double _siliconAPVNum;

#ifdef GEOMETRICDETDEBUG
  GeoHistory _parents;
  double _volume;
  double _density;
  double _weight;
  int    _copy;
  std::string _material;
  bool _fromDD;
#endif
};

#undef PoolAlloc
#endif


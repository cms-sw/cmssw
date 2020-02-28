#ifndef Geometry_TrackerNumberingBuilder_GeometricDet_H
#define Geometry_TrackerNumberingBuilder_GeometricDet_H

#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/DetId/interface/DetId.h"
#include <Math/Rotation3D.h>
#include <Math/Vector3D.h>

#include <vector>
#include <memory>
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>

class DDFilteredView;

namespace cms {
  class DDFilteredView;
}

/**
 * Composite class GeometricDet. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricDet {
public:
  using NavRange = DDExpandedView::NavRange;
  using ConstGeometricDetContainer = std::vector<GeometricDet const*>;
  using GeometricDetContainer = std::vector<GeometricDet*>;
  using RotationMatrix = ROOT::Math::Rotation3D;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >;

#ifdef PoolAlloc
  using GeoHistory = std::vector<DDExpandedNode, PoolAlloc<DDExpandedNode> >;
  using nav_type = std::vector<int, PoolAlloc<int> >;
#else
  using GeoHistory = std::vector<DDExpandedNode>;
  using nav_type = DDExpandedView::nav_type;
#endif

  using Position = Surface::PositionType;
  using Rotation = Surface::RotationType;

  //
  // more can be added; please add at the end!
  //
  typedef enum GDEnumType {
    unknown = 100,
    Tracker = 0,
    PixelBarrel = 1,
    PixelEndCap = 2,
    TIB = 3,
    TID = 4,
    TOB = 5,
    TEC = 6,
    layer = 8,
    wheel = 9,
    strng = 10,
    rod = 11,
    petal = 12,
    ring = 13,
    ladder = 14,
    mergedDet = 15,
    DetUnit = 16,
    disk = 17,
    panel = 18,
    PixelPhase1Barrel = 101,
    PixelPhase1EndCap = 102,
    PixelPhase1Disk = 117,
    OTPhase2EndCap = 204,
    OTPhase2Barrel = 205,
    OTPhase2Layer = 208,
    OTPhase2Stack = 215,
    PixelPhase2Barrel = 201,
    PixelPhase2EndCap = 202,
    OTPhase2Wheel = 209,
    PixelPhase2FullDisk = 217,
    PixelPhase2ReducedDisk = 227,
    PixelPhase2TDRDisk = 237
  } GeometricEnumType;

  /**
   * Constructors to be used when looping over DDD
   */
  GeometricDet(DDFilteredView* fv, GeometricEnumType dd);
  GeometricDet(cms::DDFilteredView* fv, GeometricEnumType dd);
  GeometricDet(const PGeometricDet::Item& onePGD, GeometricEnumType dd);

  /**
   * set or add or clear components
   */
  void setGeographicalID(DetId id) { _geographicalID = id; }
  void addComponents(GeometricDetContainer const& cont);
  void addComponents(ConstGeometricDetContainer const& cont);
  void addComponent(GeometricDet*);
  /**
   * clearComponents() only empties the container, the components are not deleted!
   */
  void clearComponents() { _container.clear(); }

  /**
   * deleteComponents() explicitly deletes the daughters
   * 
   */
  void deleteComponents();

  bool isLeaf() const { return _container.empty(); }

  GeometricDet* component(size_t index) { return const_cast<GeometricDet*>(_container[index]); }

  /**
   * Access methods
   */
  RotationMatrix const& rotation() const { return _rot; }
  Translation const& translation() const { return _trans; }
  double phi() const { return _phi; }
  double rho() const { return _rho; }

  DDSolidShape const& shape() const { return _shape; }
  GeometricEnumType type() const { return _type; }
  std::string const& name() const { return _ddname; }

  // internal representaion
  nav_type const& navType() const { return _ddd; }
  NavRange navpos() const { return NavRange(&_ddd.front(), _ddd.size()); }
  std::vector<double> const& params() const { return _params; }

  ~GeometricDet();

  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricDet it returns nothing (an empty vector)
   */
  ConstGeometricDetContainer& components() { return _container; }
  ConstGeometricDetContainer const& components() const { return _container; }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricDet is a leaf, it returns it!
   */

  ConstGeometricDetContainer deepComponents() const;
  void deepComponents(ConstGeometricDetContainer& cont) const;

  /**
   *geometricalID() returns the ID associated to the GeometricDet.
   */
  DetId geographicalID() const { return _geographicalID; }
  DetId geographicalId() const { return _geographicalID; }

  /**
   *positionBounds() returns the position in cm. 
   */
  Position positionBounds() const;

  /**
   *rotationBounds() returns the rotation matrix. 
   */
  Rotation rotationBounds() const;

  /**
   *bounds() returns the Bounds.
   */
  std::unique_ptr<Bounds> bounds() const;

  double radLength() const { return _radLength; }
  double xi() const { return _xi; }
  /**
   * The following four pix* methods only return meaningful results for pixels.
   */
  double pixROCRows() const { return _pixROCRows; }
  double pixROCCols() const { return _pixROCCols; }
  double pixROCx() const { return _pixROCx; }
  double pixROCy() const { return _pixROCy; }

  /**
   * The following two are only meaningful for the silicon tracker.
   */
  bool stereo() const { return _stereo; }
  double siliconAPVNum() const { return _siliconAPVNum; }

private:
  ConstGeometricDetContainer _container;
  Translation _trans;
  double _phi;
  double _rho;
  RotationMatrix _rot;
  DDSolidShape _shape;
  nav_type _ddd;
  std::string _ddname;
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
};

#undef PoolAlloc
#endif

#ifndef Geometry_TrackerNumberingBuilder_GeometricDet_H
#define Geometry_TrackerNumberingBuilder_GeometricDet_H

#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/types.h"

#include <DD4hep/Shapes.h>
#include <Math/Rotation3D.h>
#include <Math/Vector3D.h>

#include <vector>
#include <memory>
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
  using NavRange = std::pair<int const*, size_t>;
  using ConstGeometricDetContainer = std::vector<GeometricDet const*>;
  using GeometricDetContainer = std::vector<GeometricDet*>;
  using RotationMatrix = ROOT::Math::Rotation3D;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >;

#ifdef PoolAlloc
  using nav_type = std::vector<int, PoolAlloc<int> >;
#else
  using nav_type = std::vector<int>;
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
  void setGeographicalID(DetId id) { geographicalID_ = id; }
  void addComponents(GeometricDetContainer const& cont);
  void addComponents(ConstGeometricDetContainer const& cont);
  void addComponent(GeometricDet*);
  /**
   * clearComponents() only empties the container, the components are not deleted!
   */
  void clearComponents() { container_.clear(); }

  /**
   * deleteComponents() explicitly deletes the daughters
   * 
   */
  void deleteComponents();

  bool isLeaf() const { return container_.empty(); }

  GeometricDet* component(size_t index) { return const_cast<GeometricDet*>(container_[index]); }

  /**
   * Access methods
   */
  RotationMatrix const& rotation() const { return rot_; }
  Translation const& translation() const { return trans_; }
  double phi() const { return phi_; }
  double rho() const { return rho_; }

  // old DD
  LegacySolidShape shape() const { return cms::dd::value(cms::LegacySolidShapeMap, shape_); }
  // DD4hep
  cms::DDSolidShape shape_dd4hep() const { return shape_; }

  GeometricEnumType type() const { return type_; }
  std::string const& name() const { return ddname_; }

  // internal representaion
  // old DD
  nav_type const& navType() const { return ddd_; }
  NavRange navpos() const { return NavRange(&ddd_.front(), ddd_.size()); }

  std::vector<double> const& params() const {
    if (shape_ != cms::DDSolidShape::ddbox && shape_ != cms::DDSolidShape::ddtrap &&
        shape_ != cms::DDSolidShape::ddtubs) {
      edm::LogError("GeometricDet::params()")
          << "Called on a shape which is neither a box, a trap, nor a tub. This is not supported!";
    }
    return params_;
  }

  ~GeometricDet();

  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricDet it returns nothing (an empty vector)
   */
  ConstGeometricDetContainer& components() { return container_; }
  ConstGeometricDetContainer const& components() const { return container_; }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricDet is a leaf, it returns it!
   */

  ConstGeometricDetContainer deepComponents() const;
  void deepComponents(ConstGeometricDetContainer& cont) const;

  /**
   *geometricalID() returns the ID associated to the GeometricDet.
   */
  DetId geographicalID() const { return geographicalID_; }
  DetId geographicalId() const { return geographicalID_; }

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

  double radLength() const { return radLength_; }
  double xi() const { return xi_; }
  /**
   * The following four pix* methods only return meaningful results for pixels.
   */
  double pixROCRows() const { return pixROCRows_; }
  double pixROCCols() const { return pixROCCols_; }
  double pixROCx() const { return pixROCx_; }
  double pixROCy() const { return pixROCy_; }

  /**
   * The following two are only meaningful for the silicon tracker.
   */
  bool stereo() const { return stereo_; }
  bool isLowerSensor() const { return isLowerSensor_; }
  bool isUpperSensor() const { return isUpperSensor_; }
  double siliconAPVNum() const { return siliconAPVNum_; }

private:
  std::vector<double> computeLegacyShapeParameters(const cms::DDSolidShape& mySolidShape,
                                                   const dd4hep::Solid& mySolid) const;

  ConstGeometricDetContainer container_;
  Translation trans_;
  double phi_;
  double rho_;
  RotationMatrix rot_;
  cms::DDSolidShape shape_;
  nav_type ddd_;
  std::string ddname_;
  GeometricEnumType type_;
  std::vector<double> params_;

  DetId geographicalID_;
  double radLength_;
  double xi_;
  double pixROCRows_;
  double pixROCCols_;
  double pixROCx_;
  double pixROCy_;
  bool stereo_;
  bool isLowerSensor_;
  bool isUpperSensor_;
  double siliconAPVNum_;

  bool isFromDD4hep_;
};

#undef PoolAlloc
#endif

#ifndef Geometry_MTDNumberingBuilder_GeometricTimingDet_H
#define Geometry_MTDNumberingBuilder_GeometricTimingDet_H

#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
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
 * Composite class GeometricTimingDet. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricTimingDet {
public:
  using NavRange = std::pair<int const*, size_t>;
  using ConstGeometricTimingDetContainer = std::vector<GeometricTimingDet const*>;
  using GeometricTimingDetContainer = std::vector<GeometricTimingDet*>;
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
  using GeometricTimingEnumType = enum GTDEnumType {
    unknown = 100,
    MTD = 0,
    BTL = 1,
    BTLLayer = 2,
    BTLTray = 3,
    BTLModule = 4,
    BTLSensor = 5,
    BTLCrystal = 6,
    ETL = 7,
    ETLDisc = 8,
    ETLRing = 9,
    ETLModule = 10,
    ETLSensor = 11
  };

  /**
   * Constructors to be used when looping over DD
   */
  GeometricTimingDet(DDFilteredView* fv, GeometricTimingEnumType dd);
  GeometricTimingDet(cms::DDFilteredView* fv, GeometricTimingEnumType dd);
  GeometricTimingDet(const PGeometricTimingDet::Item& onePGD, GeometricTimingEnumType dd);

  /**
   * set or add or clear components
   */
  void setGeographicalID(DetId id) { geographicalID_ = id; }
  void addComponents(GeometricTimingDetContainer const& cont);
  void addComponents(ConstGeometricTimingDetContainer const& cont);
  void addComponent(GeometricTimingDet*);
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

  GeometricTimingDet* component(size_t index) { return const_cast<GeometricTimingDet*>(container_[index]); }

  /**
   * Access methods
   */
  RotationMatrix const& rotation() const { return rot_; }
  Translation const& translation() const { return trans_; }
  double phi() const { return phi_; }
  double rho() const { return rho_; }

  LegacySolidShape shape() const { return cms::dd::value(cms::LegacySolidShapeMap, shape_); }
  cms::DDSolidShape shape_dd4hep() const { return shape_; }
  GeometricTimingEnumType type() const { return type_; }
  std::string const& name() const { return ddname_; }
  // internal representaion
  nav_type const& navType() const { return ddd_; }
  // representation neutral interface
  NavRange navRange() const { return NavRange(&ddd_.front(), ddd_.size()); }
  // more meaningfull name (maybe)
  NavRange navpos() const { return NavRange(&ddd_.front(), ddd_.size()); }
  std::vector<double> const& params() const { return params_; }

  ~GeometricTimingDet();

  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricTimingDet it returns nothing (an empty vector)
   */
  ConstGeometricTimingDetContainer& components() { return container_; }
  ConstGeometricTimingDetContainer const& components() const { return container_; }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricTimingDet is a leaf, it returns it!
   */

  ConstGeometricTimingDetContainer deepComponents() const;
  void deepComponents(ConstGeometricTimingDetContainer& cont) const;

  /**
   *geometricalID() returns the ID associated to the GeometricTimingDet.
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
  double siliconAPVNum() const { return siliconAPVNum_; }

private:
  ConstGeometricTimingDetContainer container_;
  Translation trans_;
  double phi_;
  double rho_;
  RotationMatrix rot_;
  cms::DDSolidShape shape_;
  nav_type ddd_;
  std::string ddname_;
  GeometricTimingEnumType type_;
  std::vector<double> params_;

  DetId geographicalID_;
  double radLength_;
  double xi_;
  double pixROCRows_;
  double pixROCCols_;
  double pixROCx_;
  double pixROCy_;
  bool stereo_;
  double siliconAPVNum_;
};

#undef PoolAlloc
#endif

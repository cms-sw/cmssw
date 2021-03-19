/****************************************************************************
*
* Authors:
*	Jan Kašpar (jan.kaspar@gmail.com)
*	CMSSW developpers (based on class GeometricDet)
*
*  Rewritten + Moved out common functionalities to DetGeomDesc(Builder) by Gabrielle Hugo.
*  Migrated to DD4hep by Gabrielle Hugo and Wagner Carvalho.
*
****************************************************************************/

#ifndef Geometry_VeryForwardGeometryBuilder_DetGeomDesc
#define Geometry_VeryForwardGeometryBuilder_DetGeomDesc

#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "DataFormats/DetId/interface/DetId.h"
#include <Math/Rotation3D.h>

class CTPPSRPAlignmentCorrectionData;

/**
 * \brief Geometrical description of a sensor.
 *
 * Class resembling GeometricDet class. Slight changes were made to suit needs of the TOTEM RP description.
 * Each instance is a tree node, with geometrical information from DDD (shift, rotation, material, ...), ID and list of children nodes.
 *
 * The <b>translation</b> and <b>rotation</b> parameters are defined by <b>local-to-global</b>
 * coordinate transform. That is, if r_l is a point in local coordinate system and x_g in global,
 * then the transform reads:
 \verbatim
    x_g = rotation * x_l + translation
 \endverbatim
 *
 * July 2020: Migrated to DD4hep
 * To avoid any regression with values from XMLs / Geant4, all lengths are converted from DD4hep unit to mm.
 *
 **/

struct DiamondDimensions {
  double xHalfWidth;
  double yHalfWidth;
  double zHalfWidth;
};

class DetGeomDesc {
public:
  using Container = std::vector<DetGeomDesc*>;
  using RotationMatrix = ROOT::Math::Rotation3D;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

  // Constructor from old DD DDFilteredView
  /// \param[in] isRun2 Switch between legacy run 2-like geometry and 2021+ scenarii
  DetGeomDesc(const DDFilteredView& fv, const bool isRun2);
  // Constructor from DD4Hep DDFilteredView
  /// \param[in] isRun2 Switch between legacy run 2-like geometry and 2021+ scenarii
  DetGeomDesc(const cms::DDFilteredView& fv, const bool isRun2);

  virtual ~DetGeomDesc();

  enum CopyMode { cmWithChildren, cmWithoutChildren };
  DetGeomDesc(const DetGeomDesc& ref, CopyMode cm = cmWithChildren);

  // general info
  const std::string& name() const { return m_name; }
  int copyno() const { return m_copy; }

  // is DD4hep
  bool isDD4hep() const { return m_isDD4hep; }

  // placement info
  const Translation& translation() const { return m_trans; }  // in mm
  const RotationMatrix& rotation() const { return m_rot; }

  // shape info
  // params() is left for general access to solid shape parameters (any shape, not only box!).
  // Though, it should be used only with great care, for two reasons:
  // 1. Order of shape parameters may possibly change from a version of DD4hep to another.
  // 2. Among all parameters, those representing a length are expressed in mm (for old DD) or the DD4hep-configured unit (for DD4hep), while PPS uses mm.
  const std::vector<double>& params() const { return m_params; }  // default unit: mm for oldDD, DD4hep unit for DD4hep
  bool isABox() const { return m_isABox; }
  const DiamondDimensions& getDiamondDimensions() const {
    if (!isABox()) {
      edm::LogError("DetGeomDesc::getDiamondDimensions is not called on a box, for solid ")
          << name() << ", Id = " << geographicalID();
    }
    return m_diamondBoxParams;
  }  // in mm

  // sensor type
  const std::string& sensorType() const { return m_sensorType; }

  // ID info
  DetId geographicalID() const { return m_geographicalID; }

  // components (children) management
  const Container& components() const { return m_container; }
  float parentZPosition() const { return m_z; }  // in mm
  void addComponent(DetGeomDesc*);
  bool isLeaf() const { return m_container.empty(); }

  // alignment
  void applyAlignment(const CTPPSRPAlignmentCorrectionData&);

  void print() const;

  void invertZSign() { m_trans.SetZ(-m_trans.z()); }

private:
  void deleteComponents();      // deletes just the first daughters
  void deepDeleteComponents();  // traverses the tree and deletes all nodes.
  void clearComponents() { m_container.resize(0); }

  std::string computeNameWithNoNamespace(std::string_view nameFromView) const;
  std::vector<double> computeParameters(const cms::DDFilteredView& fv) const;
  DiamondDimensions computeDiamondDimensions(const bool isABox,
                                             const bool isDD4hep,
                                             const std::vector<double>& params) const;
  DetId computeDetID(const std::string& name,
                     const std::vector<int>& copyNos,
                     const unsigned int copyNum,
                     const bool isRun2) const;
  DetId computeDetIDFromDD4hep(const std::string& name,
                               const std::vector<int>& copyNos,
                               const unsigned int copyNum,
                               const bool isRun2) const;
  std::string computeSensorType(std::string_view name);

  std::string m_name;  // with no namespace
  int m_copy;
  bool m_isDD4hep;
  Translation m_trans;  // in mm
  RotationMatrix m_rot;
  std::vector<double> m_params;  // default unit: mm from oldDD, DD4hep unit for DD4hep
  bool m_isABox;
  DiamondDimensions m_diamondBoxParams;  // in mm
  std::string m_sensorType;
  DetId m_geographicalID;

  Container m_container;
  float m_z;  // in mm
};

struct DetGeomDescCompare {
  bool operator()(const DetGeomDesc& a, const DetGeomDesc& b) const {
    return (a.geographicalID() != b.geographicalID()
                ? a.geographicalID() < b.geographicalID()  // Sort by DetId
                // If DetIds are identical (== 0 for non-sensors), sort by name and copy number.
                : (a.name() != b.name() ? a.name() < b.name() : a.copyno() < b.copyno()));
  }
};

#endif

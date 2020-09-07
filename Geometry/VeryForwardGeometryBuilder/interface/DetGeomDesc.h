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

#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include "DataFormats/DetId/interface/DetId.h"
#include <Math/Rotation3D.h>

class DDFilteredView;
class PDetGeomDesc;
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
 * To avoid any regression with values from XMLs / Geant4, all lengths are converted from cm (DD4hep) to mm. 
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

  // Constructor from DD4Hep DDFilteredView
  DetGeomDesc(const cms::DDFilteredView& fv, const cms::DDSpecParRegistry& allSpecParSections);

  DetGeomDesc(const DetGeomDesc&);
  DetGeomDesc& operator=(const DetGeomDesc&);
  virtual ~DetGeomDesc();

  // general info
  const std::string& name() const { return m_name; }
  int copyno() const { return m_copy; }

  // placement info
  const Translation& translation() const { return m_trans; }  // in mm
  const RotationMatrix& rotation() const { return m_rot; }

  // shape info
  // params() is left for general access to solid shape parameters, but should be used
  // only with great care, for two reasons: 1. order of parameters may possibly change from
  // a version to another of DD4hep; 2. length parameters unit is cm while PPS uses mm.
  const std::vector<double>& params() const { return m_params; }  // default unit from DD4hep (cm)
  bool isABox() const { return m_isABox; }
  DiamondDimensions getDiamondDimensions() const;  // in mm

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

  bool operator<(const DetGeomDesc& other) const;
  void print() const;

private:
  void deleteComponents();      // deletes just the first daughters
  void deepDeleteComponents();  // traverses the tree and deletes all nodes.
  void clearComponents() { m_container.resize(0); }

  std::string computeNameWithNoNamespace(const std::string_view nameFromView) const;
  std::vector<double> computeParameters(const cms::DDFilteredView& fv) const;
  DetId computeDetID(const std::string& name, const std::vector<int>& copyNos, unsigned int copyNum) const;
  std::string computeSensorType(const std::string_view nameFromView,
                                const std::string& nodePath,
                                const cms::DDSpecParRegistry& allSpecParSections);

  std::string m_name;  // with no namespace
  int m_copy;
  Translation m_trans;  // in mm
  RotationMatrix m_rot;
  std::vector<double> m_params;  // default unit from DD4hep (cm)
  bool m_isABox;
  std::string m_sensorType;
  DetId m_geographicalID;

  Container m_container;
  float m_z;  // in mm
};

#endif

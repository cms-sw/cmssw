/****************************************************************************
*
* Authors:
*	Jan Kašpar (jan.kaspar@gmail.com) 
*	CMSSW developpers (based on class GeometricDet)
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
 **/

class DetGeomDesc {
public:
  using Container = std::vector<DetGeomDesc*>;
  using RotationMatrix = ROOT::Math::Rotation3D;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;

  ///Default constructors
  DetGeomDesc() {};

  ///Constructors to be used when looping over DDD
  DetGeomDesc(DDFilteredView* fv);

  ///Constructor from DD4Hep DDFilteredView
  DetGeomDesc(const cms::DDFilteredView& fv, const cms::DDSpecParRegistry& allSpecParSections);

  ///Constructor from persistent class
  DetGeomDesc(PDetGeomDesc* pd);

  /// copy constructor and assignment operator
  DetGeomDesc(const DetGeomDesc&);
  DetGeomDesc& operator=(const DetGeomDesc&);

  /// destructor
  virtual ~DetGeomDesc();

  /// ID stuff
  void setGeographicalID(DetId id) { m_geographicalID = id; }
  DetId geographicalID() const { return m_geographicalID; }

  /// access to the tree structure
  Container components() const;
  float parentZPosition() const { return m_z; }

  /// components (children) management
  void addComponent(DetGeomDesc*);
  bool isLeaf() const { return m_container.empty(); }

  /// geometry information
  RotationMatrix rotation() const { return m_rot; }
  Translation translation() const { return m_trans; }
  const std::string& name() const { return m_name; }
  std::vector<double> params() const { return m_params; }
  int copyno() const { return m_copy; }
  const std::string& sensorType() const { return m_sensorType; }
  
  /// Setters needed for use with PDetGeomDesc
  void setTranslation(double x, double y, double z) { m_trans.SetCoordinates(x,y,z); }
  void setRotation(double xx, double xy, double xz, 
                   double yx, double yy, double yz, 
                   double zx, double zy, double zz) { m_rot.SetComponents(xx, xy, xz, 
                                                                          yx, yy, yz, 
                                                                          zx, zy, zz);
  }
  void setName(std::string name) { m_name = name; }
  void setParams(std::vector<double> params) { m_params = params; }
  void setCopyno(int copy) { m_copy = copy; }
  void setParentZPosition(float z) { m_z = z; }
  void setSensorType(std::string sensorType) { m_sensorType = sensorType; }

  /// alignment
  void applyAlignment(const CTPPSRPAlignmentCorrectionData&);

private:
//  DetGeomDesc() {}
  void deleteComponents();      /// deletes just the first daughters
  void deepDeleteComponents();  /// traverses the treee and deletes all nodes.
  void clearComponents() { m_container.resize(0); }

  DetId computeDetID(const cms::DDFilteredView& fv) const;

  Container m_container;
  Translation m_trans;
  RotationMatrix m_rot;
  std::string m_name;
  std::vector<double> m_params;
  DetId m_geographicalID;
  int m_copy;
  float m_z;
  std::string m_sensorType;
};

#endif

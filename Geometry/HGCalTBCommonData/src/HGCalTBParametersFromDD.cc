#include "Geometry/HGCalTBCommonData/interface/HGCalTBParametersFromDD.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBGeomParameters.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBParameters.h"

#define EDM_ML_DEBUG
using namespace geant_units::operators;

bool HGCalTBParametersFromDD::build(const DDCompactView* cpv,
                                    HGCalTBParameters& php,
                                    const std::string& name,
                                    const std::string& namew,
                                    const std::string& namec,
                                    const std::string& namet) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBParametersFromDD (DDD)::build called with "
                                << "names " << name << ":" << namew << ":" << namec << ":" << namet;
#endif

  // Special parameters at simulation level
  std::string attribute = "Volume";
  std::string value = name;
  DDValue val(attribute, value, 0.0);
  DDSpecificsMatchesValueFilter filter{val};
  DDFilteredView fv(*cpv, filter);
  bool ok = fv.firstChild();
  HGCalGeometryMode::WaferMode mode(HGCalGeometryMode::Polyhedra);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Volume " << name << " GeometryMode ";
#endif
  if (ok) {
    DDsvalues_type sv(fv.mergedSpecifics());
    php.mode_ = HGCalGeometryMode::getGeometryMode("GeometryMode", sv);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Volume " << name << " GeometryMode " << php.mode_ << ":"
                                  << HGCalGeometryMode::Hexagon << ":" << HGCalGeometryMode::HexagonFull;
#endif
    php.levelZSide_ = 3;    // Default level for ZSide
    php.detectorType_ = 0;  // These two parameters are
    php.useSimWt_ = 1;      // energy weighting for SimHits
    std::unique_ptr<HGCalTBGeomParameters> geom = std::make_unique<HGCalTBGeomParameters>();
    if ((php.mode_ == HGCalGeometryMode::Hexagon) || (php.mode_ == HGCalGeometryMode::HexagonFull)) {
      attribute = "OnlyForHGCalNumbering";
      value = namet;
      DDValue val2(attribute, value, 0.0);
      DDSpecificsMatchesValueFilter filter2{val2};
      DDFilteredView fv2(*cpv, filter2);
      bool ok2 = fv2.firstChild();
      if (ok2) {
        DDsvalues_type sv2(fv2.mergedSpecifics());
        mode = HGCalGeometryMode::getGeometryWaferMode("WaferMode", sv2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" << HGCalGeometryMode::Polyhedra << ":"
                                      << HGCalGeometryMode::ExtrudedPolygon;
#endif
      }
    }
    php.firstLayer_ = 1;
    if (php.mode_ == HGCalGeometryMode::Hexagon) {
      // Load the SpecPars
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      // Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      // Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      // Set complete fill mode
      php.defineFull_ = false;
    } else if (php.mode_ == HGCalGeometryMode::HexagonFull) {
      // Load the SpecPars
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      // Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      // Modify some constants
      geom->loadWaferHexagon(php);
      // Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      // Set complete fill mode
      php.defineFull_ = true;
    } else {
      edm::LogError("HGCalGeom") << "Unknown Geometry type " << php.mode_ << " for HGCal " << name << ":" << namew
                                 << ":" << namec;
      throw cms::Exception("DDException")
          << "Unknown Geometry type " << php.mode_ << " for HGCal " << name << ":" << namew << ":" << namec;
    }
  } else {
    edm::LogError("HGCalGeom") << " Attribute " << val << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute " << val << " not found but needed.";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Return from HGCalTBParametersFromDD::build"
                                << " with flag " << ok;
#endif
  return ok;
}

bool HGCalTBParametersFromDD::build(const cms::DDCompactView* cpv,
                                    HGCalTBParameters& php,
                                    const std::string& name,
                                    const std::string& namew,
                                    const std::string& namec,
                                    const std::string& namet,
                                    const std::string& name2) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBParametersFromDD (DD4hep)::build called with "
                                << "names " << name << ":" << namew << ":" << namec << ":" << namet << ":" << name2;
#endif
  cms::DDVectorsMap vmap = cpv->detector()->vectors();
  const cms::DDFilter filter("Volume", name);
  cms::DDFilteredView fv((*cpv), filter);
  std::vector<std::string> tempS;
  std::vector<double> tempD;
  bool ok = fv.firstChild();
  tempS = fv.get<std::vector<std::string> >(name2, "GeometryMode");
  if (tempS.empty()) {
    tempS = fv.get<std::vector<std::string> >(name, "GeometryMode");
  }
  std::string sv = (!tempS.empty()) ? tempS[0] : "HGCalGeometryMode::Hexagon8Full";
  HGCalGeometryMode::WaferMode mode(HGCalGeometryMode::Polyhedra);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Volume " << name << " GeometryMode ";
#endif

  if (ok) {
    php.mode_ = HGCalGeometryMode::getGeometryMode(sv);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Volume " << name << " GeometryMode " << php.mode_ << ":"
                                  << HGCalGeometryMode::Hexagon << ":" << HGCalGeometryMode::HexagonFull;
#endif
    php.levelZSide_ = 3;    // Default level for ZSide
    php.detectorType_ = 0;  // These two parameters are
    php.useSimWt_ = 1;      // energy weighting for SimHits
    std::unique_ptr<HGCalTBGeomParameters> geom = std::make_unique<HGCalTBGeomParameters>();
    if ((php.mode_ == HGCalGeometryMode::Hexagon) || (php.mode_ == HGCalGeometryMode::HexagonFull)) {
      tempS = fv.get<std::vector<std::string> >(namet, "WaferMode");
      std::string sv2 = (!tempS.empty()) ? tempS[0] : "HGCalGeometryMode::Polyhedra";
      mode = HGCalGeometryMode::getGeometryWaferMode(sv2);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" << HGCalGeometryMode::Polyhedra << ":"
                                    << HGCalGeometryMode::ExtrudedPolygon;
#endif
    }
    if (php.mode_ == HGCalGeometryMode::Hexagon) {
      // Load the SpecPars
      php.firstLayer_ = 1;
      geom->loadSpecParsHexagon(fv, php, name, namew, namec, name2);
      // Load the Geometry parameters
      geom->loadGeometryHexagon(cpv, php, name, namew, namec, mode);
      // Load cell parameters
      geom->loadCellParsHexagon(vmap, php);
      // Set complete fill mode
      php.defineFull_ = false;
    } else if (php.mode_ == HGCalGeometryMode::HexagonFull) {
      // Load the SpecPars
      php.firstLayer_ = 1;
      geom->loadSpecParsHexagon(fv, php, name, namew, namec, name2);
      // Load the Geometry parameters
      geom->loadGeometryHexagon(cpv, php, name, namew, namec, mode);
      // Modify some constants
      geom->loadWaferHexagon(php);
      // Load cell parameters
      geom->loadCellParsHexagon(vmap, php);
      // Set complete fill mode
      php.defineFull_ = true;
    }
  } else {
    edm::LogError("HGCalGeom") << " Attribute Volume:" << name << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute Volume:" << name << " not found but needed.";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Return from HGCalTBParametersFromDD::build"
                                << " with flag " << ok;
#endif
  return ok;
}

double HGCalTBParametersFromDD::getDDDValue(const char* s, const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<double>& fvec = val.doubles();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "getDDDValue::Failed to get " << s << " tag.";
    }
    return fvec[0];
  } else {
    throw cms::Exception("HGCalGeom") << "getDDDValue::Failed to fetch " << s << " tag";
  }
}

std::vector<double> HGCalTBParametersFromDD::getDDDArray(const char* s, const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<double>& fvec = val.doubles();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "getDDDArray::Failed to get " << s << " tag.";
    }
    return fvec;
  } else {
    throw cms::Exception("HGCalGeom") << "getDDDArray:Failed to fetch " << s << " tag";
  }
}

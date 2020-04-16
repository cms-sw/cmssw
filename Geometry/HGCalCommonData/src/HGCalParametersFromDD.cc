#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

namespace {
  HGCalGeometryMode::GeometryMode getGeometryMode(const char* s, const DDsvalues_type& sv) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string>& fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
      }

      HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode> eparser;
      HGCalGeometryMode::GeometryMode result = (HGCalGeometryMode::GeometryMode)eparser.parseString(fvec[0]);
      return result;
    } else {
      throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag";
    }
  }
  HGCalGeometryMode::GeometryMode getGeometryMode(const std::string& s) {
    HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode> eparser;
    HGCalGeometryMode::GeometryMode result = (HGCalGeometryMode::GeometryMode)eparser.parseString(s);
    return result;
  }
  HGCalGeometryMode::WaferMode getGeometryWaferMode(const char* s, const DDsvalues_type& sv) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string>& fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
      }

      HGCalStringToEnumParser<HGCalGeometryMode::WaferMode> eparser;
      HGCalGeometryMode::WaferMode result = (HGCalGeometryMode::WaferMode)eparser.parseString(fvec[0]);
      return result;
    } else {
      throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag";
    }
  }
  HGCalGeometryMode::WaferMode getGeometryWaferMode(std::string& s) {
    HGCalStringToEnumParser<HGCalGeometryMode::WaferMode> eparser;
    HGCalGeometryMode::WaferMode result = (HGCalGeometryMode::WaferMode)eparser.parseString(s);
    return result;
  }
}  // namespace

bool HGCalParametersFromDD::build(const DDCompactView* cpv,
                                  HGCalParameters& php,
                                  const std::string& name,
                                  const std::string& namew,
                                  const std::string& namec,
                                  const std::string& namet) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersFromDD (DDD)::build called with "
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

  if (ok) {
    DDsvalues_type sv(fv.mergedSpecifics());
    php.mode_ = getGeometryMode("GeometryMode", sv);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "GeometryMode " << php.mode_ << ":" << HGCalGeometryMode::Hexagon << ":"
                                  << HGCalGeometryMode::HexagonFull << ":"
                                  << ":" << HGCalGeometryMode::Hexagon8 << ":" << HGCalGeometryMode::Hexagon8Full << ":"
                                  << ":" << HGCalGeometryMode::Trapezoid;
#endif
    php.levelZSide_ = 3;        // Default level for ZSide
    php.detectorType_ = 0;      // These two parameters are
    php.firstMixedLayer_ = -1;  // defined for post TDR geometry
    std::unique_ptr<HGCalGeomParameters> geom = std::make_unique<HGCalGeomParameters>();
    if ((php.mode_ == HGCalGeometryMode::Hexagon) || (php.mode_ == HGCalGeometryMode::HexagonFull)) {
      attribute = "OnlyForHGCalNumbering";
      value = namet;
      DDValue val2(attribute, value, 0.0);
      DDSpecificsMatchesValueFilter filter2{val2};
      DDFilteredView fv2(*cpv, filter2);
      bool ok2 = fv2.firstChild();
      if (ok2) {
        DDsvalues_type sv2(fv2.mergedSpecifics());
        mode = getGeometryWaferMode("WaferMode", sv2);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" << HGCalGeometryMode::Polyhedra << ":"
                                      << HGCalGeometryMode::ExtrudedPolygon;
#endif
      }
      php.minTileSize_ = 0;
    }
    if ((php.mode_ == HGCalGeometryMode::Hexagon8) || (php.mode_ == HGCalGeometryMode::Hexagon8Full)) {
      php.levelT_ = dbl_to_int(getDDDArray("LevelTop", sv));
      php.levelZSide_ = static_cast<int>(getDDDValue("LevelZSide", sv));
      php.nCellsFine_ = php.nCellsCoarse_ = 0;
      php.firstLayer_ = 1;
      php.firstMixedLayer_ = static_cast<int>(getDDDValue("FirstMixedLayer", sv));
      php.detectorType_ = static_cast<int>(getDDDValue("DetectorType", sv));
      php.minTileSize_ = 0;
      php.waferMaskMode_ = static_cast<int>(getDDDValue("WaferMaskMode", sv));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Top levels " << php.levelT_[0] << ":" << php.levelT_[1] << " ZSide Level "
                                    << php.levelZSide_ << " first layers " << php.firstLayer_ << ":"
                                    << php.firstMixedLayer_ << " Det Type " << php.detectorType_ << " Wafer Mask Mode "
                                    << php.waferMaskMode_;
#endif
      attribute = "OnlyForHGCalNumbering";
      value = namet;
      DDValue val2(attribute, value, 0.0);
      DDSpecificsMatchesValueFilter filter2{val2};
      DDFilteredView fv2(*cpv, filter2);
      bool ok2 = fv2.firstChild();
      if (ok2) {
        DDsvalues_type sv2(fv2.mergedSpecifics());
        mode = getGeometryWaferMode("WaferMode", sv2);
        php.nCellsFine_ = static_cast<int>(getDDDValue("NumberOfCellsFine", sv2));
        php.nCellsCoarse_ = static_cast<int>(getDDDValue("NumberOfCellsCoarse", sv2));
        php.waferSize_ = HGCalParameters::k_ScaleFromDDD * getDDDValue("WaferSize", sv2);
        php.waferThick_ = HGCalParameters::k_ScaleFromDDD * getDDDValue("WaferThickness", sv2);
        php.sensorSeparation_ = HGCalParameters::k_ScaleFromDDD * getDDDValue("SensorSeparation", sv2);
        php.mouseBite_ = HGCalParameters::k_ScaleFromDDD * getDDDValue("MouseBite", sv2);
        php.waferR_ = 0.5 * HGCalParameters::k_ScaleToDDD * php.waferSize_ / std::cos(30._deg);
        php.cellSize_.emplace_back(HGCalParameters::k_ScaleToDDD * php.waferSize_ / php.nCellsFine_);
        php.cellSize_.emplace_back(HGCalParameters::k_ScaleToDDD * php.waferSize_ / php.nCellsCoarse_);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" << HGCalGeometryMode::Polyhedra << ":"
                                      << HGCalGeometryMode::ExtrudedPolygon << " # of cells|size for fine/coarse "
                                      << php.nCellsFine_ << ":" << php.cellSize_[0] << ":" << php.nCellsCoarse_ << ":"
                                      << php.cellSize_[1] << " wafer Params " << php.waferSize_ << ":" << php.waferR_
                                      << ":" << php.waferThick_ << ":" << php.sensorSeparation_ << ":" << php.mouseBite_
                                      << ":" << php.waferR_;
#endif
        for (int k = 0; k < 2; ++k)
          getCellPosition(php, k);
      }
    }
    if (php.mode_ == HGCalGeometryMode::Hexagon) {
      // Load the SpecPars
      php.firstLayer_ = 1;
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      // Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      // Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      // Set complete fill mode
      php.defineFull_ = false;
    } else if (php.mode_ == HGCalGeometryMode::HexagonFull) {
      // Load the SpecPars
      php.firstLayer_ = 1;
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      // Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      // Modify some constants
      geom->loadWaferHexagon(php);
      // Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      // Set complete fill mode
      php.defineFull_ = true;
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8) {
      // Load the SpecPars
      geom->loadSpecParsHexagon8(fv, php);
      // Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php, 1);
      // Set complete fill mode
      php.defineFull_ = false;
      // Load wafer positions
      geom->loadWaferHexagon8(php);
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8Full) {
      // Load the SpecPars
      geom->loadSpecParsHexagon8(fv, php);
      // Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php, 1);
      // Set complete fill mode
      php.defineFull_ = true;
      // Load wafer positions
      geom->loadWaferHexagon8(php);
    } else if (php.mode_ == HGCalGeometryMode::Trapezoid) {
      // Load maximum eta & top level
      php.levelT_ = dbl_to_int(getDDDArray("LevelTop", sv));
      php.firstLayer_ = (int)(getDDDValue("FirstLayer", sv));
      php.firstMixedLayer_ = (int)(getDDDValue("FirstMixedLayer", sv));
      php.detectorType_ = (int)(getDDDValue("DetectorType", sv));
      php.waferThick_ = HGCalParameters::k_ScaleFromDDD * getDDDValue("WaferThickness", sv);
      php.minTileSize_ = HGCalParameters::k_ScaleFromDDD * getDDDValue("MinimumTileSize", sv);
      php.waferSize_ = php.waferR_ = 0;
      php.sensorSeparation_ = php.mouseBite_ = 0;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Top levels " << php.levelT_[0] << ":" << php.levelT_[1] << " first layers "
                                    << php.firstLayer_ << ":" << php.firstMixedLayer_ << " Det Type "
                                    << php.detectorType_ << "  thickenss " << php.waferThick_;
#endif
      // Load the SpecPars
      geom->loadSpecParsTrapezoid(fv, php);
      // Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php, php.firstLayer_);
      // Load cell positions
      geom->loadCellTrapezoid(php);
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
  edm::LogVerbatim("HGCalGeom") << "Return from HGCalParametersFromDD::build"
                                << " with flag " << ok;
#endif
  return ok;
}

bool HGCalParametersFromDD::build(const cms::DDCompactView* cpv,
                                  HGCalParameters& php,
                                  const std::string& name,
                                  const std::string& namew,
                                  const std::string& namec,
                                  const std::string& namet,
                                  const std::string& name2) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersFromDD (DD4Hep)::build called with "
                                << "names " << name << ":" << namew << ":" << namec << ":" << namet;
#endif
  cms::DDVectorsMap vmap = cpv->detector()->vectors();
  const cms::DDFilter filter("Volume", name);
  cms::DDFilteredView fv((*cpv), filter);
  std::vector<std::string> tempS;
  std::vector<double> tempD;
  bool ok = fv.firstChild();
  tempS = fv.get<std::vector<std::string> >(name2, "GeometryMode");
  if (tempS.empty())
    tempS = fv.get<std::vector<std::string> >(name, "GeometryMode");
  std::string sv = (!tempS.empty()) ? tempS[0] : "HGCalGeometryMode::Hexagon8Full";
  HGCalGeometryMode::WaferMode mode(HGCalGeometryMode::Polyhedra);

  if (ok) {
    php.mode_ = getGeometryMode(sv);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "GeometryMode " << php.mode_ << ":" << HGCalGeometryMode::Hexagon << ":"
                                  << HGCalGeometryMode::HexagonFull << ":"
                                  << ":" << HGCalGeometryMode::Hexagon8 << ":" << HGCalGeometryMode::Hexagon8Full << ":"
                                  << ":" << HGCalGeometryMode::Trapezoid;
#endif
    php.levelZSide_ = 3;        // Default level for ZSide
    php.detectorType_ = 0;      // These two parameters are
    php.firstMixedLayer_ = -1;  // defined for post TDR geometry
    std::unique_ptr<HGCalGeomParameters> geom = std::make_unique<HGCalGeomParameters>();
    if ((php.mode_ == HGCalGeometryMode::Hexagon) || (php.mode_ == HGCalGeometryMode::HexagonFull)) {
      tempS = fv.get<std::vector<std::string> >(namet, "WaferMode");
      std::string sv2 = (!tempS.empty()) ? tempS[0] : "HGCalGeometryMode::Polyhedra";
      mode = getGeometryWaferMode(sv2);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" << HGCalGeometryMode::Polyhedra << ":"
                                    << HGCalGeometryMode::ExtrudedPolygon;
#endif
      php.minTileSize_ = 0;
    }
    if ((php.mode_ == HGCalGeometryMode::Hexagon8) || (php.mode_ == HGCalGeometryMode::Hexagon8Full)) {
      php.levelT_ = dbl_to_int(fv.get<std::vector<double> >(name, "LevelTop"));
      tempD = fv.get<std::vector<double> >(name, "LevelZSide");
      php.levelZSide_ = static_cast<int>(tempD[0]);
      php.nCellsFine_ = php.nCellsCoarse_ = 0;
      php.firstLayer_ = 1;
      tempD = fv.get<std::vector<double> >(name, "FirstMixedLayer");
      php.firstMixedLayer_ = static_cast<int>(tempD[0]);
      tempD = fv.get<std::vector<double> >(name, "DetectorType");
      php.detectorType_ = static_cast<int>(tempD[0]);
      php.minTileSize_ = 0;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Top levels " << php.levelT_[0] << ":" << php.levelT_[1] << " ZSide Level "
                                    << php.levelZSide_ << " first layers " << php.firstLayer_ << ":"
                                    << php.firstMixedLayer_ << " Det Type " << php.detectorType_;
#endif

      tempS = fv.get<std::vector<std::string> >(namet, "WaferMode");
      std::string sv2 = (!tempS.empty()) ? tempS[0] : "HGCalGeometryMode::ExtrudedPolygon";
      mode = getGeometryWaferMode(sv2);
      tempD = fv.get<std::vector<double> >(namet, "NumberOfCellsFine");
      php.nCellsFine_ = static_cast<int>(tempD[0]);
      tempD = fv.get<std::vector<double> >(namet, "NumberOfCellsCoarse");
      php.nCellsCoarse_ = static_cast<int>(tempD[0]);
      tempD = fv.get<std::vector<double> >(namet, "WaferSize");
      php.waferSize_ = HGCalParameters::k_ScaleFromDD4Hep * tempD[0];
      tempD = fv.get<std::vector<double> >(namet, "WaferThickness");
      php.waferThick_ = HGCalParameters::k_ScaleFromDD4Hep * tempD[0];
      tempD = fv.get<std::vector<double> >(namet, "SensorSeparation");
      php.sensorSeparation_ = HGCalParameters::k_ScaleFromDD4Hep * tempD[0];
      tempD = fv.get<std::vector<double> >(namet, "MouseBite");
      php.mouseBite_ = HGCalParameters::k_ScaleFromDD4Hep * tempD[0];
      php.waferR_ = 0.5 * HGCalParameters::k_ScaleToDDD * php.waferSize_ / std::cos(30._deg);
      php.cellSize_.emplace_back(HGCalParameters::k_ScaleToDDD * php.waferSize_ / php.nCellsFine_);
      php.cellSize_.emplace_back(HGCalParameters::k_ScaleToDDD * php.waferSize_ / php.nCellsCoarse_);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" << HGCalGeometryMode::Polyhedra << ":"
                                    << HGCalGeometryMode::ExtrudedPolygon << " # of cells|size for fine/coarse "
                                    << php.nCellsFine_ << ":" << php.cellSize_[0] << ":" << php.nCellsCoarse_ << ":"
                                    << php.cellSize_[1] << " wafer Params " << php.waferSize_ << ":" << php.waferR_
                                    << ":" << php.waferThick_ << ":" << php.sensorSeparation_ << ":" << php.mouseBite_
                                    << ":" << php.waferR_;
#endif
      for (int k = 0; k < 2; ++k)
        getCellPosition(php, k);
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
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8) {
      // Load the SpecPars
      geom->loadSpecParsHexagon8(fv, vmap, php, name);
      // Load Geometry parameters
      geom->loadGeometryHexagon8(cpv, php, name, 1);
      // Set complete fill mode
      php.defineFull_ = false;
      // Load wafer positions
      geom->loadWaferHexagon8(php);
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8Full) {
      // Load the SpecPars
      geom->loadSpecParsHexagon8(fv, vmap, php, name);
      // Load Geometry parameters
      geom->loadGeometryHexagon8(cpv, php, name, 1);
      // Set complete fill mode
      php.defineFull_ = true;
      // Load wafer positions
      geom->loadWaferHexagon8(php);
    } else if (php.mode_ == HGCalGeometryMode::Trapezoid) {
      // Load maximum eta & top level
      php.levelT_ = dbl_to_int(fv.get<std::vector<double> >(name, "LevelTop"));
      tempD = fv.get<std::vector<double> >(name, "LevelZSide");
      php.levelZSide_ = static_cast<int>(tempD[0]);
      php.nCellsFine_ = php.nCellsCoarse_ = 0;
      tempD = fv.get<std::vector<double> >(name, "FirstLayer");
      php.firstLayer_ = static_cast<int>(tempD[0]);
      tempD = fv.get<std::vector<double> >(name, "FirstMixedLayer");
      php.firstMixedLayer_ = static_cast<int>(tempD[0]);
      tempD = fv.get<std::vector<double> >(name, "DetectorType");
      php.detectorType_ = static_cast<int>(tempD[0]);
      tempD = fv.get<std::vector<double> >(name, "WaferThickness");
      php.waferThick_ = HGCalParameters::k_ScaleFromDD4Hep * tempD[0];
      tempD = fv.get<std::vector<double> >(name, "MinimumTileSize");
      php.minTileSize_ = HGCalParameters::k_ScaleFromDD4Hep * tempD[0];
      php.waferSize_ = php.waferR_ = 0;
      php.sensorSeparation_ = php.mouseBite_ = 0;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Top levels " << php.levelT_[0] << ":" << php.levelT_[1] << " first layers "
                                    << php.firstLayer_ << ":" << php.firstMixedLayer_ << " Det Type "
                                    << php.detectorType_ << "  thickenss " << php.waferThick_ << " min tile size "
                                    << php.minTileSize_;
#endif
      // Load the SpecPars
      geom->loadSpecParsTrapezoid(fv, vmap, php, name);
      // Load Geometry parameters
      geom->loadGeometryHexagon8(cpv, php, name, php.firstLayer_);
      // Load cell positions
      geom->loadCellTrapezoid(php);
    } else {
      edm::LogError("HGCalGeom") << "Unknown Geometry type " << php.mode_ << " for HGCal " << name << ":" << namew
                                 << ":" << namec;
      throw cms::Exception("DDException")
          << "Unknown Geometry type " << php.mode_ << " for HGCal " << name << ":" << namew << ":" << namec;
    }
  } else {
    edm::LogError("HGCalGeom") << " Attribute Volume:" << name << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute Volume:" << name << " not found but needed.";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Return from HGCalParametersFromDD::build"
                                << " with flag " << ok;
#endif
  return ok;
}

void HGCalParametersFromDD::getCellPosition(HGCalParameters& php, int type) {
  if (type == 1) {
    php.cellCoarseX_.clear();
    php.cellCoarseY_.clear();
  } else {
    php.cellFineX_.clear();
    php.cellFineY_.clear();
  }
  HGCalParameters::wafer_map cellIndex;
#ifdef EDM_ML_DEBUG
  std::vector<int> indtypes;
#endif
  int N = (type == 1) ? php.nCellsCoarse_ : php.nCellsFine_;
  double R = php.waferSize_ / (3 * N);
  double r = 0.5 * R * sqrt(3.0);
  int n2 = N / 2;
  int ipos(0);
  for (int u = 0; u < 2 * N; ++u) {
    for (int v = 0; v < 2 * N; ++v) {
      if (((v - u) < N) && (u - v) <= N) {
        double yp = (u - 0.5 * v - n2) * 2 * r;
        double xp = (1.5 * (v - N) + 1.0) * R;
        int id = v * 100 + u;
#ifdef EDM_ML_DEBUG
        indtypes.emplace_back(id);
#endif
        if (type == 1) {
          php.cellCoarseX_.emplace_back(xp);
          php.cellCoarseY_.emplace_back(yp);
        } else {
          php.cellFineX_.emplace_back(xp);
          php.cellFineY_.emplace_back(yp);
        }
        cellIndex[id] = ipos;
        ++ipos;
      }
    }
  }
  if (type == 1)
    php.cellCoarseIndex_ = cellIndex;
  else
    php.cellFineIndex_ = cellIndex;
#ifdef EDM_ML_DEBUG
  if (type == 1) {
    edm::LogVerbatim("HGCalGeom") << "CellPosition for  type " << type << " for " << php.cellCoarseX_.size()
                                  << " cells";
    for (unsigned int k = 0; k < php.cellCoarseX_.size(); ++k) {
      int id = indtypes[k];
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] ID " << id << ":" << php.cellCoarseIndex_[id] << " X "
                                    << php.cellCoarseX_[k] << " Y " << php.cellCoarseY_[k];
    }
  } else {
    edm::LogVerbatim("HGCalGeom") << "CellPosition for  type " << type << " for " << php.cellFineX_.size() << " cells";
    for (unsigned int k = 0; k < php.cellFineX_.size(); ++k) {
      int id = indtypes[k];
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] ID " << id << ":" << php.cellFineIndex_[k] << " X "
                                    << php.cellFineX_[k] << " Y " << php.cellFineY_[k];
    }
  }
#endif
}

double HGCalParametersFromDD::getDDDValue(const char* s, const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<double>& fvec = val.doubles();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
    }
    return fvec[0];
  } else {
    throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag";
  }
}

std::vector<double> HGCalParametersFromDD::getDDDArray(const char* s, const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<double>& fvec = val.doubles();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
    }
    return fvec;
  } else {
    throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag";
  }
}

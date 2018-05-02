#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

namespace {
  HGCalGeometryMode::GeometryMode getGeometryMode(const char* s, 
						  const DDsvalues_type & sv) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string> & fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
      }

      HGCalStringToEnumParser<HGCalGeometryMode::GeometryMode> eparser;
      HGCalGeometryMode::GeometryMode result = (HGCalGeometryMode::GeometryMode) eparser.parseString(fvec[0]);
      return result;
    } else {
      throw cms::Exception("HGCalGeom") << "Failed to get "<< s << " tag";
    }
  }
  HGCalGeometryMode::WaferMode getGeometryWaferMode(const char* s,
						    const DDsvalues_type & sv){
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string> & fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
      }

      HGCalStringToEnumParser<HGCalGeometryMode::WaferMode> eparser;
      HGCalGeometryMode::WaferMode result = (HGCalGeometryMode::WaferMode) eparser.parseString(fvec[0]);
      return result;
    } else {
      throw cms::Exception("HGCalGeom") << "Failed to get "<< s << " tag";
    }
  }
}

bool HGCalParametersFromDD::build(const DDCompactView* cpv,
				  HGCalParameters& php, 
				  const std::string& name,
				  const std::string& namew, 
				  const std::string& namec) {

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersFromDD::build called with "
				<< "names " << name << ":" << namew << ":" 
				<< namec;
#endif

  //Special parameters at simulation level
  std::string attribute = "Volume"; 
  std::string value     = name;
  DDValue val(attribute, value, 0.0);
  DDSpecificsMatchesValueFilter filter{val};
  DDFilteredView fv(*cpv,filter);
  bool ok = fv.firstChild();
  HGCalGeometryMode::WaferMode mode(HGCalGeometryMode::Polyhedra);

  if (ok) {
    DDsvalues_type sv(fv.mergedSpecifics());
    php.mode_ = getGeometryMode("GeometryMode", sv);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "GeometryMode " << php.mode_ 
				  << ":" << HGCalGeometryMode::Hexagon << ":" 
				  << HGCalGeometryMode::HexagonFull << ":"
				  << ":" << HGCalGeometryMode::Hexagon8 << ":"
				  << HGCalGeometryMode::Hexagon8Full << ":"
				  << ":" << HGCalGeometryMode::Trapezoid;
#endif
    HGCalGeomParameters *geom = new HGCalGeomParameters();
    if ((php.mode_ == HGCalGeometryMode::Hexagon) ||
	(php.mode_ == HGCalGeometryMode::HexagonFull)) {
      attribute  = "OnlyForHGCalNumbering";
      value      = "HGCal";
      DDValue val2(attribute, value, 0.0);
      DDSpecificsMatchesValueFilter filter2{val2};
      DDFilteredView fv2(*cpv,filter2);
      bool ok2 = fv2.firstChild();
      if (ok2) {
	DDsvalues_type sv2(fv2.mergedSpecifics());
	mode = getGeometryWaferMode("WaferMode", sv2);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" 
				      << HGCalGeometryMode::Polyhedra << ":" 
				      << HGCalGeometryMode::ExtrudedPolygon;
#endif
      }
    }
    if ((php.mode_ == HGCalGeometryMode::Hexagon8) ||
	(php.mode_ == HGCalGeometryMode::Hexagon8Full)) {
      attribute  = "OnlyForHGCalNumbering";
      value      = "HGCal";
      DDValue val2(attribute, value, 0.0);
      DDSpecificsMatchesValueFilter filter2{val2};
      DDFilteredView fv2(*cpv,filter2);
      bool ok2 = fv2.firstChild();
      if (ok2) {
	DDsvalues_type sv2(fv2.mergedSpecifics());
	mode = getGeometryWaferMode("WaferMode", sv2);
	php.nCellsFine_       = (int)(getDDDValue("NumberOfCellsFine",  sv2));
	php.nCellsCoarse_     = (int)(getDDDValue("NumberOfCellsCoarse",sv2));
	php.waferSize_        = getDDDValue("WaferSize", sv2);
	php.waferThick_       = getDDDValue("WaferThickness", sv2);
	php.sensorSeparation_ = getDDDValue("SensorSeparation", sv2);
	php.mouseBite_        = getDDDValue("MouseBite", sv2);
	php.waferR_           = php.waferSize_/std::sqrt(3.0);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" 
				      << HGCalGeometryMode::Polyhedra << ":" 
				      << HGCalGeometryMode::ExtrudedPolygon
				      << "# of cells in fine/coarse wafers "
				      << php.nCellsFine_ << ":" 
				      << php.nCellsCoarse_ << " wafer Params "
				      << php.waferSize_ << ":"
				      << php.waferR_ << ":"
				      << php.waferThick_ << ":"
				      << php.sensorSeparation_ << ":"
				      << php.mouseBite_;
#endif
      }
    }
    if (php.mode_ == HGCalGeometryMode::Hexagon) {
      //Load the SpecPars
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      //Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      //Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      //Set complete fill mode
      php.defineFull_ = false;
    } else if (php.mode_ == HGCalGeometryMode::HexagonFull) {
      //Load the SpecPars
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      //Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      //Modify some constants
      geom->loadWaferHexagon(php);
      //Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      //Set complete fill mode
      php.defineFull_ = true;
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8) {
      //Load the SpecPars
      geom->loadSpecParsHexagon8(fv, php);
      //Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php);
      //Load wafer positions
      geom->loadWaferHexagon8(php);
      //Set complete fill mode
      php.defineFull_ = false;
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8Full) {
      //Load the SpecPars
      geom->loadSpecParsHexagon8(fv, php);
      //Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php);
      //Load wafer positions
      geom->loadWaferHexagon8(php);
      //Set complete fill mode
      php.defineFull_ = true;
    } else if (php.mode_ == HGCalGeometryMode::Trapezoid) {
      //Load maximum eta
      php.etaMaxBH_ = getDDDValue("etaMaxBH", sv);
      //Load the SpecPars
      geom->loadSpecParsTrapezoid(fv, php);
      //Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php);
    } else {
      edm::LogError("HGCalGeom") << "Unknown Geometry type " << php.mode_
				 << " for HGCal " << name << ":" << namew
				 << ":" << namec;
      throw cms::Exception("DDException") 
	<< "Unknown Geometry type " << php.mode_ << " for HGCal " << name 
	<< ":" << namew << ":" << namec;
    }
  } else {
      edm::LogError("HGCalGeom") << " Attribute " << val
				 << " not found but needed.";
      throw cms::Exception("DDException") << "Attribute " << val
					  << " not found but needed.";
  }

  edm::LogInfo("HGCalGeom") << "Return from HGCalParametersFromDD::build with "
			    << ok;
  return ok;
}

double HGCalParametersFromDD::getDDDValue(const char* s, 
					  const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<double> & fvec = val.doubles();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
    }
    return fvec[0];
  } else {
    throw cms::Exception("HGCalGeom") << "Failed to get "<< s << " tag";
  }
}

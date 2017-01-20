#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

namespace {
  int getGeometryMode(const char* s, const DDsvalues_type & sv) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string> & fvec = val.strings();
      if (fvec.size() == 0) {
        throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
      }

      int result(-1);
      HGCalStringToEnumParser<HGCalGeometryMode> eparser;
      HGCalGeometryMode mode = (HGCalGeometryMode) eparser.parseString(fvec[0]);
      result = (int)(mode);
      return result;
    } else {
      throw cms::Exception("HGCalGeom") << "Failed to get "<< s << " tag.";
    }
  }
}

bool HGCalParametersFromDD::build(const DDCompactView* cpv,
				  HGCalParameters& php, 
				  const std::string& name,
				  const std::string& namew, 
				  const std::string& namec) {

#ifdef EDM_ML_DEBUG
  std::cout << "HGCalParametersFromDD::build called with names " << name << ":"
	    << namew << ":" << namec << std::endl;
#endif

  //Special parameters at simulation level
  std::string attribute = "Volume"; 
  std::string value     = name;
  DDValue val(attribute, value, 0.0);
  DDSpecificsFilter filter;
  filter.setCriteria(val, DDCompOp::equals);
  DDFilteredView fv(*cpv);
  fv.addFilter(filter);
  bool ok = fv.firstChild();

  if (ok) {
    DDsvalues_type sv(fv.mergedSpecifics());
    php.mode_ = getGeometryMode("GeometryMode", sv);
    HGCalGeomParameters *geom = new HGCalGeomParameters();
    if (php.mode_ == static_cast<int> (HGCalGeometryMode::Square)) {
      //Load the SpecPars
      geom->loadSpecParsSquare(fv, php);
      //Load the Geometry parameters
      geom->loadGeometrySquare(fv, php, name);
    } else if (php.mode_ == static_cast<int> (HGCalGeometryMode::Hexagon)) {
      //Load the SpecPars
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      //Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec);
      //Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
    } else if (php.mode_ == static_cast<int> (HGCalGeometryMode::HexagonFull)){
      //Load the SpecPars
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      //Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec);
      //Modify some constants
      geom->loadWaferHexagon(php);
      //Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
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

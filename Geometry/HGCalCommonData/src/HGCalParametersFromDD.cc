#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "CondFormats/GeometryObjects/interface/HGCalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Base/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

namespace {
  int getGeometryMode(const char* s, const DDsvalues_type & sv) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string> & fvec = val.strings();
      if (fvec.size() == 0) {
        throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
      }

      int result(-1);
      StringToEnumParser<HGCalGeometryMode::GeometryMode> eparser;
      HGCalGeometryMode::GeometryMode mode = (HGCalGeometryMode::GeometryMode) eparser.parseString(fvec[0]);
      result = (int)(mode);
      return result;
    } else {
      throw cms::Exception("HGCalGeom") << "Failed to get "<< s << " tag.";
    }
  }
}

bool HGCalParametersFromDD::build(const DDCompactView* cpv,
				  HGCalParameters& php, 
				  const std::string& name) {

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
    if (php.mode_ == HGCalGeometryMode::Square) {
      //Load the SpecPars
      geom->loadSpecParsSquare(fv, php);
      //Load the Geometry parameters
      geom->loadGeometrySquare(fv, php, name);
    }
  } else {
    throw cms::Exception("HGCalGeom") << "Not found "<< attribute.c_str() << " but needed.";
  }

  edm::LogInfo("HGCalGeom") << "Return from HGCalParametersFromDD::build with "
			    << ok;
  return ok;
}

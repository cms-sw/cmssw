#include "Geometry/HGCalCommonData/interface/FastTimeParametersFromDD.h"
#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

bool FastTimeParametersFromDD::build(const DDCompactView* cpv,
				     FastTimeParameters& php, 
				     const std::string& name,
				     const int type) { 

#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeParametersFromDD::build called with names " << name 
	    << " and type " << type << std::endl;
#endif

  //Special parameters at simulation level
  std::string attribute = "Volume"; 
  const std::string& value     = name;
  DDValue val(attribute, value, 0.0);
  DDSpecificsMatchesValueFilter filter{val};
  DDFilteredView fv(*cpv,filter);
  bool ok = fv.firstChild();

  if (ok) {
    DDsvalues_type sv(fv.mergedSpecifics());
    std::vector<double> temp;
    if (type == 1) {
      php.geomParBarrel_ = getDDDArray("geomParsB", sv);
      temp               = getDDDArray("numberZB", sv);
      php.nZBarrel_      = (int)(temp[0]);
      temp               = getDDDArray("numberPhiB", sv);
      php.nPhiBarrel_    = (int)(temp[0]);
#ifdef EDM_ML_DEBUG
      std::cout << "Barrel Parameters: " << php.nZBarrel_ << ":" 
		<< php.nPhiBarrel_ << ":" << php.geomParBarrel_[0] << ":"
		<< php.geomParBarrel_[1] << std::endl;
#endif
    } else if (type == 2) {
      php.geomParEndcap_ = getDDDArray("geomParsE", sv);
      temp               = getDDDArray("numberEtaE", sv);
      php.nEtaEndcap_    = (int)(temp[0]);
      temp               = getDDDArray("numberPhiE", sv);
      php.nPhiEndcap_    = (int)(temp[0]);
#ifdef EDM_ML_DEBUG
      std::cout << "Endcap Parameters: " << php.nEtaEndcap_ << ":" 
		<< php.nPhiEndcap_ << ":" << php.geomParEndcap_[0] << ":"
		<< php.geomParEndcap_[1] << ":" << php.geomParEndcap_[2]
		<< std::endl;
#endif
    } else {
      edm::LogWarning("HGCalGeom") << "Unknown Geometry type " << type
				   << " for FastTiming " << name;
    }
  } else {
      edm::LogError("HGCalGeom") << " Attribute " << val
				 << " not found but needed.";
      throw cms::Exception("DDException") << "Attribute " << val
					  << " not found but needed.";
  }
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeParametersFromDD::Returns with flag " << ok 
	    << " for " << name  << " and type " << type << std::endl;
#endif
  return ok;
}

std::vector<double> FastTimeParametersFromDD::getDDDArray(const std::string & str, 
							  const DDsvalues_type & sv) {

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters : # of " << str
				 << " bins " << nval << " < 1 ==> illegal";
      throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
    }
    return fvec;
  } else {
    edm::LogError("HGCalGeom") << "HGCalGeomParameters: cannot get array "
			       << str;
    throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
  }
}

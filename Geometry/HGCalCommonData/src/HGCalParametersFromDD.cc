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
				  const std::string& namec,
				  const std::string& namet) {

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalParametersFromDD::build called with "
				<< "names " << name << ":" << namew << ":" 
				<< namec << ":" << namet;
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
    php.levelZSide_      = 3;       // Default level for ZSide
    php.detectorType_    = 0;       // These two parameters are
    php.firstMixedLayer_ =-1;       // defined for post TDR geometry
    HGCalGeomParameters *geom = new HGCalGeomParameters();
    if ((php.mode_ == HGCalGeometryMode::Hexagon) ||
	(php.mode_ == HGCalGeometryMode::HexagonFull)) {
      attribute  = "OnlyForHGCalNumbering";
      value      = namet;
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
      php.levelT_          = dbl_to_int(getDDDArray("LevelTop",sv));
      php.levelZSide_      = (int)(getDDDValue("LevelZSide",sv));
      php.nCellsFine_      = php.nCellsCoarse_ = 0;
      php.firstLayer_      = 1;
      php.firstMixedLayer_ = (int)(getDDDValue("FirstMixedLayer", sv));
      php.detectorType_    = (int)(getDDDValue("DetectorType", sv));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Top levels " << php.levelT_[0] << ":" 
				    << php.levelT_[1] << " ZSide Level "
				    << php.levelZSide_ << " first layers "
				    << php.firstLayer_ << ":"
				    << php.firstMixedLayer_ << " Det Type "
				    << php.detectorType_;
#endif
      attribute   = "OnlyForHGCalNumbering";
      value       = namet;
      DDValue val2(attribute, value, 0.0);
      DDSpecificsMatchesValueFilter filter2{val2};
      DDFilteredView fv2(*cpv,filter2);
      bool ok2 = fv2.firstChild();
      if (ok2) {
	DDsvalues_type sv2(fv2.mergedSpecifics());
	mode = getGeometryWaferMode("WaferMode", sv2);
	php.nCellsFine_       = (int)(getDDDValue("NumberOfCellsFine",  sv2));
	php.nCellsCoarse_     = (int)(getDDDValue("NumberOfCellsCoarse",sv2));
	php.waferSize_        = HGCalParameters::k_ScaleFromDDD*getDDDValue("WaferSize", sv2);
	php.waferThick_       = HGCalParameters::k_ScaleFromDDD*getDDDValue("WaferThickness", sv2);
	php.sensorSeparation_ = HGCalParameters::k_ScaleFromDDD*getDDDValue("SensorSeparation", sv2);
	php.mouseBite_        = HGCalParameters::k_ScaleFromDDD*getDDDValue("MouseBite", sv2);
	php.waferR_           = 0.5*HGCalParameters::k_ScaleToDDD*php.waferSize_/std::cos(30.0*CLHEP::deg);
	php.etaMinBH_         = 0;
	php.cellSize_.emplace_back(HGCalParameters::k_ScaleToDDD*php.waferSize_/php.nCellsFine_);
	php.cellSize_.emplace_back(HGCalParameters::k_ScaleToDDD*php.waferSize_/php.nCellsCoarse_);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HGCalGeom") << "WaferMode " << mode << ":" 
				      << HGCalGeometryMode::Polyhedra << ":" 
				      << HGCalGeometryMode::ExtrudedPolygon
				      << " # of cells|size for fine/coarse "
				      << php.nCellsFine_ << ":" 
				      << php.cellSize_[0] << ":"
				      << php.nCellsCoarse_ << ":"
				      << php.cellSize_[1] << " wafer Params "
				      << php.waferSize_ << ":"
				      << php.waferR_ << ":"
				      << php.waferThick_ << ":"
				      << php.sensorSeparation_ << ":"
				      << php.mouseBite_ << ":" << php.waferR_;
#endif
	for (int k=0; k<2; ++k) getCellPosition(php, k);
      }
    }
    if (php.mode_ == HGCalGeometryMode::Hexagon) {
      //Load the SpecPars
      php.firstLayer_ = 1;
      geom->loadSpecParsHexagon(fv, php, cpv, namew, namec);
      //Load the Geometry parameters
      geom->loadGeometryHexagon(fv, php, name, cpv, namew, namec, mode);
      //Load cell parameters
      geom->loadCellParsHexagon(cpv, php);
      //Set complete fill mode
      php.defineFull_ = false;
    } else if (php.mode_ == HGCalGeometryMode::HexagonFull) {
      //Load the SpecPars
      php.firstLayer_ = 1;
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
      geom->loadGeometryHexagon8(fv, php, 1);
      //Set complete fill mode
      php.defineFull_ = false;
      //Load wafer positions
      geom->loadWaferHexagon8(php);
    } else if (php.mode_ == HGCalGeometryMode::Hexagon8Full) {
      //Load the SpecPars
      geom->loadSpecParsHexagon8(fv, php);
      //Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php, 1);
      //Set complete fill mode
      php.defineFull_ = true;
      //Load wafer positions
      geom->loadWaferHexagon8(php);
    } else if (php.mode_ == HGCalGeometryMode::Trapezoid) {
      //Load maximum eta & top level
      php.etaMinBH_        = getDDDValue("etaMinBH", sv);
      php.levelT_          = dbl_to_int(getDDDArray("LevelTop",sv));
      php.firstLayer_      = (int)(getDDDValue("FirstLayer", sv));
      php.firstMixedLayer_ = (int)(getDDDValue("FirstMixedLayer", sv));
      php.detectorType_    = (int)(getDDDValue("DetectorType", sv));
      php.waferThick_      = HGCalParameters::k_ScaleFromDDD*getDDDValue("WaferThickness", sv);
      php.waferSize_       = php.waferR_          = 0;
      php.sensorSeparation_= php.mouseBite_       = 0;
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "Top levels " << php.levelT_[0] << ":" 
				    << php.levelT_[1] << " EtaMinBH "
				    << php.etaMinBH_ << " first layers "
				    << php.firstLayer_ << ":" 
				    << php.firstMixedLayer_ << " Det Type "
				    << php.detectorType_ << "  thickenss "
				    << php.waferThick_;
#endif
      //Load the SpecPars
      geom->loadSpecParsTrapezoid(fv, php);
      //Load Geometry parameters
      geom->loadGeometryHexagon8(fv, php, php.firstLayer_);
      //Load cell positions
      geom->loadCellTrapezoid(php);
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
  HGCalParameters::wafer_map  cellIndex;
#ifdef EDM_ML_DEBUG
  std::vector<int> indtypes;
#endif    
  int    N  = (type == 1) ? php.nCellsCoarse_ : php.nCellsFine_;
  double R  = php.waferSize_/(3*N);
  double r  = 0.5*R*sqrt(3.0);
  int    n2 = N/2;
  int    ipos(0);
  for (int u=0; u<2*N; ++u) {
    for (int v=0; v<2*N; ++v) {
      if (((v-u) < N) && (u-v) <= N) {
        double yp = (u-0.5*v-n2)*2*r;
        double xp = (1.5*(v-N)+1.0)*R;
	int    id = v*100 + u;
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
  if (type == 1) php.cellCoarseIndex_ = cellIndex;
  else           php.cellFineIndex_   = cellIndex;
#ifdef EDM_ML_DEBUG
  if (type == 1) {
    edm::LogVerbatim("HGCalGeom") << "CellPosition for  type " << type
				  << " for " << php.cellCoarseX_.size()
				  << " cells";
    for (unsigned int k=0; k<php.cellCoarseX_.size(); ++k) {
      int id = indtypes[k];
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] ID " << id << ":"
				    << php.cellCoarseIndex_[id] << " X "
                                    << php.cellCoarseX_[k] << " Y "
                                    << php.cellCoarseY_[k];
    }
  } else {
    edm::LogVerbatim("HGCalGeom") << "CellPosition for  type " << type
				  << " for " << php.cellFineX_.size()
				  << " cells";
    for (unsigned int k=0; k<php.cellCoarseX_.size(); ++k) {
      int id = indtypes[k];
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] ID " << id << ":"
				    << php.cellFineIndex_[k] << " X "
                                    << php.cellFineX_[k] << " Y "
                                    << php.cellFineY_[k];
    }
  }
#endif
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

std::vector<double> HGCalParametersFromDD::getDDDArray(const char* s, 
						       const DDsvalues_type& sv) {
  DDValue val(s);
  if (DDfetch(&sv, val)) {
    const std::vector<double> & fvec = val.doubles();
    if (fvec.empty()) {
      throw cms::Exception("HGCalGeom") << "Failed to get " << s << " tag.";
    }
    return fvec;
  } else {
    throw cms::Exception("HGCalGeom") << "Failed to get "<< s << " tag";
  }
}

#include "Geometry/HcalCommonData/interface/CaloSimParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

template <typename T>
void myPrint(std::string value, const std::vector<T>& vec) {
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << vec.size() << " entries for " << value << ":";
  unsigned int i(0);
  for (const auto& e : vec) {
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << e;
    ++i;
  }
}

bool CaloSimParametersFromDD::build(const DDCompactView* cpv, CaloSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom")
      << "Inside CaloSimParametersFromDD::build(const DDCompactView*, CaloSimulationParameters&)";
#endif
  // Get the names
  std::string attribute = "ReadOutName";
  std::string name = "CaloHitsTk";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, name, 0)};
  DDFilteredView fv(*cpv, filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  php.caloNames_ = getNames("Calorimeter", sv, false);
  php.levels_ = getNumbers("Levels", sv, false);
  php.neighbours_ = getNumbers("Neighbours", sv, false);
  php.insideNames_ = getNames("Inside", sv, false);
  php.insideLevel_ = getNumbers("InsideLevel", sv, false);
  php.fCaloNames_ = getNames("FineCalorimeter", sv, true);
  php.fLevels_ = getNumbers("FineLevels", sv, true);

  return this->buildParameters(php);
}

bool CaloSimParametersFromDD::build(const cms::DDCompactView* cpv, CaloSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom")
      << "Inside CaloSimParametersFromDD::build(const cms::DDCompactView*, CaloSimulationParameters&)";
#endif
  // Get the names
  cms::DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());
  php.caloNames_ = fv.get<std::vector<std::string> >("calo", "Calorimeter");
  php.levels_ = dbl_to_int(fv.get<std::vector<double> >("calo", "Levels"));
  php.neighbours_ = dbl_to_int(fv.get<std::vector<double> >("calo", "Neighbours"));
  php.insideNames_ = fv.get<std::vector<std::string> >("calo", "Inside");
  php.insideLevel_ = dbl_to_int(fv.get<std::vector<double> >("calo", "InsideLevel"));
  php.fCaloNames_ = fv.get<std::vector<std::string> >("calo", "FineCalorimeter");
  php.fLevels_ = dbl_to_int(fv.get<std::vector<double> >("calo", "FineLevels"));

  return this->buildParameters(php);
}

bool CaloSimParametersFromDD::buildParameters(const CaloSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  myPrint("Calorimeter", php.caloNames_);
  myPrint("Levels", php.levels_);
  myPrint("Neighbours", php.neighbours_);
  myPrint("Inside", php.insideNames_);
  myPrint("InsideLevel", php.insideLevel_);
  myPrint("FineCalorimeter", php.fCaloNames_);
  myPrint("FineLevels", php.fLevels_);
#endif

  if (php.caloNames_.size() < php.neighbours_.size()) {
    edm::LogError("HCalGeom") << "CaloSimParametersFromDD: # of Calorimeter bins " << php.caloNames_.size()
                              << " does not match with " << php.neighbours_.size() << " ==> illegal ";
    throw cms::Exception("Unknown", "CaloSimParametersFromDD")
        << "Calorimeter array size does not match with size of neighbours\n";
  }

  return true;
}

std::vector<std::string> CaloSimParametersFromDD::getNames(const std::string& str,
                                                           const DDsvalues_type& sv,
                                                           bool ignore) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD::getNames called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << value;
#endif
    const std::vector<std::string>& fvec = value.strings();
    int nval = fvec.size();
    if ((nval < 1) && (!ignore)) {
      edm::LogError("HCalGeom") << "CaloSimParametersFromDD: # of " << str << " bins " << nval << " < 1 ==> illegal ";
      throw cms::Exception("Unknown", "CaloSimParametersFromDD") << "nval < 2 for array " << str << "\n";
    }

    return fvec;
  } else if (ignore) {
    std::vector<std::string> fvec;
    return fvec;
  } else {
    edm::LogError("HCalGeom") << "CaloSimParametersFromDD: cannot get array " << str;
    throw cms::Exception("Unknown", "CaloSimParametersFromDD") << "cannot get array " << str << "\n";
  }
}

std::vector<int> CaloSimParametersFromDD::getNumbers(const std::string& str, const DDsvalues_type& sv, bool ignore) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD::getNumbers called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << value;
#endif
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if ((nval < 1) && (!ignore)) {
      edm::LogError("HCalGeom") << "CaloSimParametersFromDD: # of " << str << " bins " << nval << " < 1 ==> illegal ";
      throw cms::Exception("Unknown", "CaloSimParametersFromDD") << "nval < 2 for array " << str << "\n";
    }
    return dbl_to_int(fvec);
  } else if (ignore) {
    std::vector<int> fvec;
    return fvec;
  } else {
    edm::LogError("HCalGeom") << "CaloSimParametersFromDD: cannot get array " << str;
    throw cms::Exception("Unknown", "CaloSimParametersFromDD") << "cannot get array " << str << "\n";
  }
}

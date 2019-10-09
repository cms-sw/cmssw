#include "Geometry/HcalCommonData/interface/CaloSimParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/CaloSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

CaloSimParametersFromDD::CaloSimParametersFromDD(bool fromDD4Hep) : fromDD4Hep_(fromDD4Hep) { 
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: initialized with fromDD4Hep = " << fromDD4Hep_;
#endif
}

bool CaloSimParametersFromDD::build(const DDCompactView* cpv, CaloSimulationParameters& php) {
  // Get the names
  std::string attribute = "ReadOutName";
  std::string name = "CaloHitsTk";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, name, 0)};
  DDFilteredView fv(*cpv, filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  std::string value = "Calorimeter";
  php.caloNames_ = getNames(value, sv, false);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.caloNames_.size() << " entries for " << value << ":";
  for (unsigned int i = 0; i < php.caloNames_.size(); i++)
    edm::LogVerbatim("HCaloGeom") << " (" << i << ") " << php.caloNames_[i];
#endif

  value = "Levels";
  php.levels_ = getNumbers(value, sv, false);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.levels_.size() << " entries for " << value << ":";
  for (unsigned int i = 0; i < php.levels_.size(); i++)
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << php.levels_[i];
#endif

  value = "Neighbours";
  php.neighbours_ = getNumbers(value, sv, false);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.neighbours_.size() << " entries for " << value << ":";
  for (unsigned int i = 0; i < php.neighbours_.size(); i++)
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << php.neighbours_[i];
#endif

  value = "Inside";
  php.insideNames_ = getNames(value, sv, false);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.insideNames_.size() << " entries for " << value << ":";
  for (unsigned int i = 0; i < php.insideNames_.size(); i++)
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << php.insideNames_[i];
#endif

  value = "InsideLevel";
  php.insideLevel_ = getNumbers(value, sv, false);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.insideLevel_.size() << " ebtries for " << value << ":";
  for (unsigned int i = 0; i < php.insideLevel_.size(); i++)
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << php.insideLevel_[i];
#endif

  value = "FineCalorimeter";
  php.fCaloNames_ = getNames(value, sv, true);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.fCaloNames_.size() << " entries for " << value << ":";
  for (unsigned int i = 0; i < php.fCaloNames_.size(); i++)
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << php.fCaloNames_[i];
#endif

  value = "FineLevels";
  php.fLevels_ = getNumbers(value, sv, true);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "CaloSimParametersFromDD: " << php.fLevels_.size() << " entries for " << value << ":";
  for (unsigned int i = 0; i < php.fLevels_.size(); i++)
    edm::LogVerbatim("HCalGeom") << " (" << i << ") " << php.fLevels_[i];
#endif

  if (php.caloNames_.size() < php.neighbours_.size()) {
    edm::LogError("HCalGeom") << "CaloSimParametersFromDD: # of Calorimeter bins " << php.caloNames_.size()
                             << " does not match with " << php.neighbours_.size() << " ==> illegal ";
    throw cms::Exception("Unknown", "CaloSimParametersFromDD")
        << "Calorimeter array size does not match with size of neighbours\n";
  }

  return true;
}

std::vector<std::string> CaloSimParametersFromDD::getNames(const std::string& str, const DDsvalues_type& sv, bool ignore) {
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

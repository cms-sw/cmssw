#include "Geometry/HcalCommonData/interface/HcalSimParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

#define EDM_ML_DEBUG

using namespace geant_units::operators;

bool HcalSimParametersFromDD::build(const DDCompactView* cpv, HcalSimulationParameters& php) {
  // Parameters for the fibers
  std::string attribute = "Volume";
  std::string value = "HF";
  DDSpecificsMatchesValueFilter filter1{DDValue(attribute, value, 0)};
  DDFilteredView fv1(*cpv, filter1);

  // Names of sensitive volumes for HF
  php.hfNames_ = getNames(fv1);
  int nb(-1);
#ifdef EDM_ML_DEBUG
  nb = static_cast<int>(php.hfNames_.size());
  std::stringstream ss0;
  for (int it = 0; it < nb; it++) {
    if (it / 10 * 10 == it) {
      ss0 << "\n";
    }
    ss0 << " [" << it << "] " << php.hfNames_[it];
  }
  edm::LogVerbatim("HCalGeom") << "HFNames: " << nb << ": " << ss0.str();
#endif

  bool dodet = fv1.firstChild();
  if (dodet) {
    DDsvalues_type sv(fv1.mergedSpecifics());

    // The level positions
    nb = -1;
    php.hfLevels_ = dbl_to_int(getDDDArray("Levels", sv, nb));
#ifdef EDM_ML_DEBUG
    std::stringstream ss0;
    for (int it = 0; it < nb; it++) {
      if (it / 10 * 10 == it) {
        ss0 << "\n";
      }
      ss0 << " [" << it << "] " << php.hfLevels_[it];
    }
    edm::LogVerbatim("HCalGeom") << "HF Volume Levels: " << nb << " hfLevels: " << ss0.str();
#endif

    // Attenuation length
    nb = -1;
    php.attenuationLength_ = getDDDArray("attl", sv, nb);
#ifdef EDM_ML_DEBUG
    std::stringstream ss1;
    for (int it = 0; it < nb; it++) {
      if (it / 10 * 10 == it) {
        ss1 << "\n";
      }
      ss1 << "  " << convertMmToCm(php.attenuationLength_[it]);
    }
    edm::LogVerbatim("HCalGeom") << "AttenuationLength: " << nb << " attL(1/cm): " << ss1.str();
#endif

    // Limits on Lambda
    nb = 2;
    php.lambdaLimits_ = dbl_to_int(getDDDArray("lambLim", sv, nb));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "Limits on lambda " << php.lambdaLimits_[0] << " and " << php.lambdaLimits_[1];
#endif

    // Fibre Lengths
    nb = 0;
    php.longFiberLength_ = getDDDArray("LongFL", sv, nb);
#ifdef EDM_ML_DEBUG
    std::stringstream ss2;
    for (int it = 0; it < nb; it++) {
      if (it / 10 * 10 == it) {
        ss2 << "\n";
      }
      ss2 << "  " << convertMmToCm(php.longFiberLength_[it]);
    }
    edm::LogVerbatim("HCalGeom") << nb << " Long Fibre Length(cm):" << ss2.str();
#endif

    nb = 0;
    php.shortFiberLength_ = getDDDArray("ShortFL", sv, nb);
#ifdef EDM_ML_DEBUG
    std::stringstream ss3;
    for (int it = 0; it < nb; it++) {
      if (it / 10 * 10 == it) {
        ss3 << "\n";
      }
      ss3 << "  " << convertMmToCm(php.shortFiberLength_[it]);
    }
    edm::LogVerbatim("HCalGeom") << nb << " Short Fibre Length(cm):" << ss3.str();
#endif

  } else {
    throw cms::Exception("HcalSimParametersFromDD") << "Not found " << value << " for " << attribute << " but needed.";
  }

  //Parameters for the PMT
  value = "HFPMT";
  DDSpecificsMatchesValueFilter filter2{DDValue(attribute, value, 0)};
  DDFilteredView fv2(*cpv, filter2);
  if (fv2.firstChild()) {
    DDsvalues_type sv(fv2.mergedSpecifics());
    int nb = -1;
    std::vector<double> neta = getDDDArray("indexPMTR", sv, nb);
    for (unsigned int ii = 0; ii < neta.size(); ii++) {
      int index = static_cast<int>(neta[ii]);
      int ir = -1, ifib = -1;
      if (index >= 0) {
        ir = index / 10;
        ifib = index % 10;
      }
      php.pmtRight_.push_back(ir);
      php.pmtFiberRight_.push_back(ifib);
    }
    nb = -1;
    neta = getDDDArray("indexPMTL", sv, nb);
    for (unsigned int ii = 0; ii < neta.size(); ii++) {
      int index = static_cast<int>(neta[ii]);
      int ir = -1, ifib = -1;
      if (index >= 0) {
        ir = index / 10;
        ifib = index % 10;
      }
      php.pmtLeft_.push_back(ir);
      php.pmtFiberLeft_.push_back(ifib);
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "HcalSimParameters: gets the Index matches for " << neta.size() << " PMTs";
    for (unsigned int ii = 0; ii < neta.size(); ii++) {
      edm::LogVerbatim("HCalGeom") << "rIndexR[" << ii << "] = " << php.pmtRight_[ii] << " fibreR[" << ii
                                   << "] = " << php.pmtFiberRight_[ii] << " rIndexL[" << ii
                                   << "] = " << php.pmtLeft_[ii] << " fibreL[" << ii << "] = " << php.pmtFiberLeft_[ii];
    }
#endif
  } else {
    throw cms::Exception("HcalSimParametersFromDD") << "Not found " << value << " for " << attribute << " but needed.";
  }

  //Names of special volumes (HFFibre, HFPMT, HFFibreBundles)
  fillNameVector(cpv, attribute, "HFFibre", php.hfFibreNames_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfFibreNames_.size() << " names of HFFibre";
  for (unsigned int k = 0; k < php.hfFibreNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfFibreNames_[k];
#endif
  fillNameVector(cpv, attribute, "HFPMT", php.hfPMTNames_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfPMTNames_.size() << " names of HFPMT";
  for (unsigned int k = 0; k < php.hfPMTNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfPMTNames_[k];
#endif
  fillNameVector(cpv, attribute, "HFFibreBundleStraight", php.hfFibreStraightNames_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfFibreStraightNames_.size()
                               << " names of HFFibreBundleStraight";
  for (unsigned int k = 0; k < php.hfFibreStraightNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfFibreStraightNames_[k];
#endif
  fillNameVector(cpv, attribute, "HFFibreBundleConical", php.hfFibreConicalNames_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfFibreConicalNames_.size()
                               << " names of FibreBundleConical";
  for (unsigned int k = 0; k < php.hfFibreConicalNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfFibreConicalNames_[k];
#endif

  // HCal materials
  attribute = "OnlyForHcalSimNumbering";
  DDSpecificsHasNamedValueFilter filter3{attribute};
  DDFilteredView fv3(*cpv, filter3);
  dodet = fv3.firstChild();

  while (dodet) {
    const DDLogicalPart& log = fv3.logicalPart();
    if (!isItHF(log.name().name(), php)) {
      bool notIn = true;
      for (unsigned int i = 0; i < php.hcalMaterialNames_.size(); ++i) {
        if (!strcmp(php.hcalMaterialNames_[i].c_str(), log.material().name().name().c_str())) {
          notIn = false;
          break;
        }
      }
      if (notIn)
        php.hcalMaterialNames_.push_back(log.material().name().name());
    }
    dodet = fv2.next();
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hcalMaterialNames_.size() << " names of HCAL materials";
  for (unsigned int k = 0; k < php.hcalMaterialNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hcalMaterialNames_[k];
#endif

  return true;
}

void HcalSimParametersFromDD::fillNameVector(const DDCompactView* cpv,
                                             const std::string& attribute,
                                             const std::string& value,
                                             std::vector<std::string>& lvnames) {
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0)};
  DDFilteredView fv(*cpv, filter);
  lvnames = getNames(fv);
}

bool HcalSimParametersFromDD::isItHF(const std::string& name, const HcalSimulationParameters& php) {
  for (auto nam : php.hfNames_)
    if (name == nam)
      return true;
  for (auto nam : php.hfFibreNames_)
    if (name == nam)
      return true;
  for (auto nam : php.hfPMTNames_)
    if (name == nam)
      return true;
  for (auto nam : php.hfFibreStraightNames_)
    if (name == nam)
      return true;
  for (auto nam : php.hfFibreConicalNames_)
    if (name == nam)
      return true;

  return false;
}

std::vector<std::string> HcalSimParametersFromDD::getNames(DDFilteredView& fv) {
  std::vector<std::string> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart& log = fv.logicalPart();
    bool ok = true;

    for (unsigned int i = 0; i < tmp.size(); ++i) {
      if (!strcmp(tmp[i].c_str(), log.name().name().c_str())) {
        ok = false;
        break;
      }
    }
    if (ok)
      tmp.push_back(log.name().name());
    dodet = fv.next();
  }
  return tmp;
}

std::vector<double> HcalSimParametersFromDD::getDDDArray(const std::string& str, const DDsvalues_type& sv, int& nmin) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalSimParametersFromDD::getDDDArray called for " << str << " with nMin " << nmin;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << value;
#endif
    const std::vector<double>& fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        edm::LogError("HCalGeom") << "# of " << str << " bins " << nval << " < " << nmin << " ==> illegal";
        throw cms::Exception("HcalSimParametersFromDD") << "nval < nmin for array " << str << "\n";
      }
    } else {
      if (nval < 1 && nmin != 0) {
        edm::LogError("HCalGeom") << "# of " << str << " bins " << nval << " < 1 ==> illegal (nmin=" << nmin << ")";
        throw cms::Exception("HcalSimParametersFromDD") << "nval < 1 for array " << str << "\n";
      }
    }
    nmin = nval;
    return fvec;
  } else {
    if (nmin != 0) {
      edm::LogError("HCalGeom") << "Cannot get array " << str;
      throw cms::Exception("HcalSimParametersFromDD") << "cannot get array " << str << "\n";
    } else {
      std::vector<double> fvec;
      return fvec;
    }
  }
}

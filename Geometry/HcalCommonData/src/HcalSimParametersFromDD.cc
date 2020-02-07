#include "Geometry/HcalCommonData/interface/HcalSimParametersFromDD.h"
#include "CondFormats/GeometryObjects/interface/HcalSimulationParameters.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DataFormats/Math/interface/GeantUnits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

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

  bool dodet = fv1.firstChild();
  if (dodet) {
    DDsvalues_type sv(fv1.mergedSpecifics());

    // The level positions
    nb = -1;
    php.hfLevels_ = dbl_to_int(getDDDArray("Levels", sv, nb));

    // Attenuation length
    nb = -1;
    php.attenuationLength_ = getDDDArray("attl", sv, nb);

    // Limits on Lambda
    nb = 2;
    php.lambdaLimits_ = dbl_to_int(getDDDArray("lambLim", sv, nb));

    // Fibre Lengths
    nb = 0;
    php.longFiberLength_ = getDDDArray("LongFL", sv, nb);

    nb = 0;
    php.shortFiberLength_ = getDDDArray("ShortFL", sv, nb);

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
    fillPMTs(neta, false, php);
    nb = -1;
    neta = getDDDArray("indexPMTL", sv, nb);
    fillPMTs(neta, true, php);
  } else {
    throw cms::Exception("HcalSimParametersFromDD") << "Not found " << value << " for " << attribute << " but needed.";
  }

  //Names of special volumes (HFFibre, HFPMT, HFFibreBundles)
  fillNameVector(cpv, attribute, "HFFibre", php.hfFibreNames_);
  fillNameVector(cpv, attribute, "HFPMT", php.hfPMTNames_);
  fillNameVector(cpv, attribute, "HFFibreBundleStraight", php.hfFibreStraightNames_);
  fillNameVector(cpv, attribute, "HFFibreBundleConical", php.hfFibreConicalNames_);

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

  return buildParameters(php);
}

bool HcalSimParametersFromDD::build(const cms::DDCompactView* cpv, HcalSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom")
      << "Inside HcalSimParametersFromDD::build(const cms::DDCompactView*, HcalSimulationParameters&)";
#endif

  // Parameters for the fibers
  fillNameVector(cpv, "HF", php.hfNames_);

  // The level positions
  cms::DDFilteredView fv1(cpv->detector(), cpv->detector()->worldVolume());
  php.hfLevels_ = dbl_to_int(fv1.get<std::vector<double> >("hf", "Levels"));

  // Attenuation length
  static const double cminv2mminv = 0.1;
  php.attenuationLength_ = fv1.get<std::vector<double> >("hf", "attl");
  std::for_each(php.attenuationLength_.begin(), php.attenuationLength_.end(), [](double& n) { n *= cminv2mminv; });

  // Limits on Lambda
  php.lambdaLimits_ = dbl_to_int(fv1.get<std::vector<double> >("hf", "lambLim"));

  // Fibre Lengths
  php.longFiberLength_ = fv1.get<std::vector<double> >("hf", "LongFL");
  std::for_each(php.longFiberLength_.begin(), php.longFiberLength_.end(), [](double& n) { n = convertCmToMm(n); });
  php.shortFiberLength_ = fv1.get<std::vector<double> >("hf", "ShortFL");
  std::for_each(php.shortFiberLength_.begin(), php.shortFiberLength_.end(), [](double& n) { n = convertCmToMm(n); });

  //Parameters for the PMT
  std::vector<double> neta = fv1.get<std::vector<double> >("hfpmt", "indexPMTR");
  fillPMTs(neta, false, php);
  neta = fv1.get<std::vector<double> >("hfpmt", "indexPMTL");
  fillPMTs(neta, true, php);

  //Names of special volumes (HFFibre, HFPMT, HFFibreBundles)
  fillNameVector(cpv, "HFFibre", php.hfFibreNames_);
  fillNameVector(cpv, "HFPMT", php.hfPMTNames_);
  fillNameVector(cpv, "HFFibreBundleStraight", php.hfFibreStraightNames_);
  fillNameVector(cpv, "HFFibreBundleConical", php.hfFibreConicalNames_);

  // HCal materials
  cms::DDFilteredView fv2(cpv->detector(), cpv->detector()->worldVolume());
  cms::DDSpecParRefs ref2;
  const cms::DDSpecParRegistry& par2 = cpv->specpars();
  par2.filter(ref2, "OnlyForHcalSimNumbering", "HCAL");
  fv2.mergedSpecifics(ref2);

  while (fv2.firstChild()) {
    const std::string matName{cms::dd::noNamespace(fv2.materialName()).data(),
                              cms::dd::noNamespace(fv2.materialName()).size()};
    std::vector<int> copy = fv2.copyNos();
    // idet = 3 for HB and 4 for HE (convention in the ddalgo code for HB/HE)
    int idet = (copy.size() > 1) ? (copy[1] / 1000) : 0;
    if (((idet == 3) || (idet == 4)) &&
        std::find(std::begin(php.hcalMaterialNames_), std::end(php.hcalMaterialNames_), matName) ==
            std::end(php.hcalMaterialNames_)) {
      php.hcalMaterialNames_.emplace_back(matName);
    }
  };
  return buildParameters(php);
}

bool HcalSimParametersFromDD::buildParameters(const HcalSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  std::stringstream ss0;
  for (unsigned int it = 0; it < php.hfNames_.size(); it++) {
    if (it / 10 * 10 == it)
      ss0 << "\n";
    ss0 << " [" << it << "] " << php.hfNames_[it];
  }
  edm::LogVerbatim("HCalGeom") << "HFNames: " << php.hfNames_.size() << ": " << ss0.str();

  std::stringstream ss1;
  for (unsigned int it = 0; it < php.hfLevels_.size(); it++) {
    if (it / 10 * 10 == it)
      ss1 << "\n";
    ss1 << " [" << it << "] " << php.hfLevels_[it];
  }
  edm::LogVerbatim("HCalGeom") << "HF Volume Levels: " << php.hfLevels_.size() << " hfLevels: " << ss1.str();

  std::stringstream ss2;
  for (unsigned int it = 0; it < php.attenuationLength_.size(); it++) {
    if (it / 10 * 10 == it)
      ss2 << "\n";
    ss2 << "  " << convertMmToCm(php.attenuationLength_[it]);
  }
  edm::LogVerbatim("HCalGeom") << "AttenuationLength: " << php.attenuationLength_.size()
                               << " attL(1/cm): " << ss2.str();

  std::stringstream ss3;
  for (unsigned int it = 0; it < php.lambdaLimits_.size(); it++) {
    if (it / 10 * 10 == it)
      ss3 << "\n";
    ss3 << "  " << php.lambdaLimits_[it];
  }
  edm::LogVerbatim("HCalGeom") << php.lambdaLimits_.size() << " Limits on lambda " << ss3.str();

  std::stringstream ss4;
  for (unsigned int it = 0; it < php.longFiberLength_.size(); it++) {
    if (it / 10 * 10 == it)
      ss4 << "\n";
    ss4 << "  " << convertMmToCm(php.longFiberLength_[it]);
  }
  edm::LogVerbatim("HCalGeom") << php.longFiberLength_.size() << " Long Fibre Length(cm):" << ss4.str();

  std::stringstream ss5;
  for (unsigned int it = 0; it < php.shortFiberLength_.size(); it++) {
    if (it / 10 * 10 == it)
      ss5 << "\n";
    ss5 << "  " << convertMmToCm(php.shortFiberLength_[it]);
  }
  edm::LogVerbatim("HCalGeom") << php.shortFiberLength_.size() << " Short Fibre Length(cm):" << ss5.str();

  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: gets the Index matches for " << php.pmtRight_.size() << " PMTs";
  for (unsigned int ii = 0; ii < php.pmtRight_.size(); ii++)
    edm::LogVerbatim("HCalGeom") << "rIndexR[" << ii << "] = " << php.pmtRight_[ii] << " fibreR[" << ii
                                 << "] = " << php.pmtFiberRight_[ii] << " rIndexL[" << ii << "] = " << php.pmtLeft_[ii]
                                 << " fibreL[" << ii << "] = " << php.pmtFiberLeft_[ii];

  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfFibreNames_.size() << " names of HFFibre";
  for (unsigned int k = 0; k < php.hfFibreNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfFibreNames_[k];

  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfPMTNames_.size() << " names of HFPMT";
  for (unsigned int k = 0; k < php.hfPMTNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfPMTNames_[k];

  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfFibreStraightNames_.size()
                               << " names of HFFibreBundleStraight";
  for (unsigned int k = 0; k < php.hfFibreStraightNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfFibreStraightNames_[k];

  edm::LogVerbatim("HCalGeom") << "HcalSimParameters: " << php.hfFibreConicalNames_.size()
                               << " names of FibreBundleConical";
  for (unsigned int k = 0; k < php.hfFibreConicalNames_.size(); ++k)
    edm::LogVerbatim("HCalGeom") << "[" << k << "] " << php.hfFibreConicalNames_[k];

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

void HcalSimParametersFromDD::fillNameVector(const cms::DDCompactView* cpv,
                                             const std::string& value,
                                             std::vector<std::string>& lvnames) {
  cms::DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());
  cms::DDSpecParRefs refs;
  const cms::DDSpecParRegistry& mypar = cpv->specpars();
  mypar.filter(refs, "Volume", value);
  fv.mergedSpecifics(refs);
  lvnames = getNames(fv);
}

void HcalSimParametersFromDD::fillPMTs(const std::vector<double>& neta, bool lOrR, HcalSimulationParameters& php) {
  for (unsigned int ii = 0; ii < neta.size(); ii++) {
    int index = static_cast<int>(neta[ii]);
    int ir = -1, ifib = -1;
    if (index >= 0) {
      ir = index / 10;
      ifib = index % 10;
    }
    if (lOrR) {
      php.pmtLeft_.push_back(ir);
      php.pmtFiberLeft_.push_back(ifib);
    } else {
      php.pmtRight_.push_back(ir);
      php.pmtFiberRight_.push_back(ifib);
    }
  }
}

bool HcalSimParametersFromDD::isItHF(const std::string& name, const HcalSimulationParameters& php) {
  if (std::find(std::begin(php.hfNames_), std::end(php.hfNames_), name) != std::end(php.hfNames_))
    return true;
  if (std::find(std::begin(php.hfFibreNames_), std::end(php.hfFibreNames_), name) != std::end(php.hfFibreNames_))
    return true;
  if (std::find(std::begin(php.hfPMTNames_), std::end(php.hfPMTNames_), name) != std::end(php.hfPMTNames_))
    return true;
  if (std::find(std::begin(php.hfFibreStraightNames_), std::end(php.hfFibreStraightNames_), name) !=
      std::end(php.hfFibreStraightNames_))
    return true;
  if (std::find(std::begin(php.hfFibreConicalNames_), std::end(php.hfFibreConicalNames_), name) !=
      std::end(php.hfFibreConicalNames_))
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

std::vector<std::string> HcalSimParametersFromDD::getNames(cms::DDFilteredView& fv) {
  std::vector<std::string> tmp;
  while (fv.firstChild()) {
    if (std::find(std::begin(tmp), std::end(tmp), fv.name()) == std::end(tmp))
      tmp.emplace_back(fv.name());
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

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
  bool dodet = fv1.firstChild();

  if (dodet) {
    DDsvalues_type sv(fv1.mergedSpecifics());

    // Attenuation length
    int nb = -1;
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

  return true;
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

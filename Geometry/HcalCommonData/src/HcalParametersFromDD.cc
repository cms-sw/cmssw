#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"
#include "Geometry/HcalCommonData/interface/HcalGeomParameters.h"
#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

namespace {
  int getTopologyMode(const char* s, const DDsvalues_type& sv, bool type) {
    DDValue val(s);
    if (DDfetch(&sv, val)) {
      const std::vector<std::string>& fvec = val.strings();
      if (fvec.empty()) {
        throw cms::Exception("HcalParametersFromDD") << "Failed to get " << s << " tag.";
      }

      int result(-1);
      if (type) {
        StringToEnumParser<HcalTopologyMode::Mode> eparser;
        HcalTopologyMode::Mode mode = (HcalTopologyMode::Mode)eparser.parseString(fvec[0]);
        result = static_cast<int>(mode);
      } else {
        StringToEnumParser<HcalTopologyMode::TriggerMode> eparser;
        HcalTopologyMode::TriggerMode mode = (HcalTopologyMode::TriggerMode)eparser.parseString(fvec[0]);
        result = static_cast<int>(mode);
      }
      return result;
    } else {
      throw cms::Exception("HcalParametersFromDD") << "Failed to get " << s << " tag.";
    }
  }
}  // namespace

bool HcalParametersFromDD::build(const DDCompactView* cpv, HcalParameters& php) {
  //Special parameters at simulation level
  std::string attribute = "OnlyForHcalSimNumbering";
  DDSpecificsHasNamedValueFilter filter1{attribute};
  DDFilteredView fv1(*cpv, filter1);
  bool ok = fv1.firstChild();

  const int nEtaMax = 100;

  if (ok) {
    std::unique_ptr<HcalGeomParameters> geom = std::make_unique<HcalGeomParameters>();
    geom->loadGeometry(fv1, php);
    php.modHB = geom->getModHalfHBHE(0);
    php.modHE = geom->getModHalfHBHE(1);
    php.dzVcal = geom->getConstDzHF();
    geom->getConstRHO(php.rHO);

    php.phioff = DDVectorGetter::get("phioff");
    php.etaTable = DDVectorGetter::get("etaTable");
    php.rTable = DDVectorGetter::get("rTable");
    php.phibin = DDVectorGetter::get("phibin");
    php.phitable = DDVectorGetter::get("phitable");
    for (unsigned int i = 1; i <= nEtaMax; ++i) {
      std::stringstream sstm;
      sstm << "layerGroupSimEta" << i;
      std::string tempName = sstm.str();
      if (DDVectorGetter::check(tempName)) {
        HcalParameters::LayerItem layerGroupEta;
        layerGroupEta.layer = i;
        layerGroupEta.layerGroup = dbl_to_int(DDVectorGetter::get(tempName));
        php.layerGroupEtaSim.emplace_back(layerGroupEta);
      }
    }
    php.etaMin = dbl_to_int(DDVectorGetter::get("etaMin"));
    php.etaMax = dbl_to_int(DDVectorGetter::get("etaMax"));
    php.etaMin[0] = 1;
    if (php.etaMax[1] >= php.etaMin[1])
      php.etaMax[1] = static_cast<int>(php.etaTable.size()) - 1;
    php.etaMax[2] = php.etaMin[2] + static_cast<int>(php.rTable.size()) - 2;
    php.etaRange = DDVectorGetter::get("etaRange");
    php.gparHF = DDVectorGetter::get("gparHF");
    php.noff = dbl_to_int(DDVectorGetter::get("noff"));
    php.Layer0Wt = DDVectorGetter::get("Layer0Wt");
    php.HBGains = DDVectorGetter::get("HBGains");
    php.HBShift = dbl_to_int(DDVectorGetter::get("HBShift"));
    php.HEGains = DDVectorGetter::get("HEGains");
    php.HEShift = dbl_to_int(DDVectorGetter::get("HEShift"));
    php.HFGains = DDVectorGetter::get("HFGains");
    php.HFShift = dbl_to_int(DDVectorGetter::get("HFShift"));
    php.maxDepth = dbl_to_int(DDVectorGetter::get("MaxDepth"));
  } else {
    throw cms::Exception("HcalParametersFromDD") << "Not found " << attribute.c_str() << " but needed.";
  }
  for (unsigned int i = 0; i < php.rTable.size(); ++i) {
    unsigned int k = php.rTable.size() - i - 1;
    php.etaTableHF.emplace_back(-log(tan(0.5 * atan(php.rTable[k] / php.gparHF[4]))));
  }
  //Special parameters at reconstruction level
  attribute = "OnlyForHcalRecNumbering";
  DDSpecificsHasNamedValueFilter filter2{attribute};
  DDFilteredView fv2(*cpv, filter2);
  ok = fv2.firstChild();
  if (ok) {
    DDsvalues_type sv(fv2.mergedSpecifics());
    int topoMode = getTopologyMode("TopologyMode", sv, true);
    int trigMode = getTopologyMode("TriggerMode", sv, false);
    php.topologyMode = ((trigMode & 0xFF) << 8) | (topoMode & 0xFF);
    php.etagroup = dbl_to_int(DDVectorGetter::get("etagroup"));
    php.phigroup = dbl_to_int(DDVectorGetter::get("phigroup"));
    for (unsigned int i = 1; i <= nEtaMax; ++i) {
      std::stringstream sstm;
      sstm << "layerGroupRecEta" << i;
      std::string tempName = sstm.str();
      if (DDVectorGetter::check(tempName)) {
        HcalParameters::LayerItem layerGroupEta;
        layerGroupEta.layer = i;
        layerGroupEta.layerGroup = dbl_to_int(DDVectorGetter::get(tempName));
        php.layerGroupEtaRec.emplace_back(layerGroupEta);
      }
    }
  } else {
    throw cms::Exception("HcalParametersFromDD") << "Not found " << attribute.c_str() << " but needed.";
  }

  return build(php);
}

bool HcalParametersFromDD::build(const HcalParameters& php) {
#ifdef EDM_ML_DEBUG
  int i(0);
  std::stringstream ss0;
  for (unsigned int it = 0; it < php.maxDepth.size(); it++) {
    if (it / 10 * 10 == it) {
      ss0 << "\n";
    }
    ss0 << " [" << it << "] " << php.maxDepth[it];
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::maxDepth: " << php.maxDepth.size() << ": " << ss0.str();
  std::stringstream ss1;
  for (unsigned int it = 0; it < php.modHB.size(); it++) {
    if (it / 10 * 10 == it) {
      ss1 << "\n";
    }
    ss1 << " [" << it << "] " << php.modHB[it];
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::modHB: " << php.modHB.size() << ": " << ss1.str();
  std::stringstream ss2;
  for (unsigned int it = 0; it < php.modHE.size(); it++) {
    if (it / 10 * 10 == it) {
      ss2 << "\n";
    }
    ss2 << " [" << it << "] " << php.modHE[it];
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::modHE: " << php.modHE.size() << ": " << ss2.str();
  std::stringstream ss3;
  for (unsigned int it = 0; it < php.phioff.size(); it++) {
    if (it / 10 * 10 == it) {
      ss3 << "\n";
    }
    ss3 << " [" << it << "] " << convertRadToDeg(php.phioff[it]);
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::phiOff: " << php.phioff.size() << ": " << ss3.str();
  std::stringstream ss4;
  for (unsigned int it = 0; it < php.etaTable.size(); it++) {
    if (it / 10 * 10 == it) {
      ss4 << "\n";
    }
    ss4 << " [" << it << "] " << php.etaTable[it];
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::etaTable: " << php.etaTable.size() << ": " << ss4.str();
  std::stringstream ss5;
  for (unsigned int it = 0; it < php.rTable.size(); it++) {
    if (it / 10 * 10 == it) {
      ss5 << "\n";
    }
    ss5 << " [" << it << "] " << convertMmToCm(php.rTable[it]);
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::rTable: " << php.rTable.size() << ": " << ss5.str();
  std::stringstream ss6;
  for (unsigned int it = 0; it < php.phibin.size(); it++) {
    if (it / 10 * 10 == it) {
      ss6 << "\n";
    }
    ss6 << " [" << it << "] " << convertRadToDeg(php.phibin[it]);
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD:phibin: " << php.phibin.size() << ": " << ss6.str();
  std::stringstream ss7;
  for (unsigned int it = 0; it < php.phitable.size(); it++) {
    if (it / 10 * 10 == it) {
      ss7 << "\n";
    }
    ss7 << " [" << it << "] " << convertRadToDeg(php.phitable[it]);
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD:phitable: " << php.phitable.size() << ": " << ss7.str();
  std::cout << "HcalParametersFromDD: " << php.layerGroupEtaSim.size() << " layerGroupEtaSim blocks" << std::endl;
  std::vector<int>::const_iterator kt;
  std::vector<double>::const_iterator it;
  for (unsigned int k = 0; k < php.layerGroupEtaSim.size(); ++k) {
    std::cout << "layerGroupEtaSim[" << k << "] Layer " << php.layerGroupEtaSim[k].layer;
    for (kt = php.layerGroupEtaSim[k].layerGroup.begin(), i = 0; kt != php.layerGroupEtaSim[k].layerGroup.end(); ++kt)
      std::cout << " " << ++i << ":" << (*kt);
    std::cout << std::endl;
  }
  std::cout << "HcalParametersFromDD: " << php.etaMin.size() << " etaMin values";
  for (kt = php.etaMin.begin(), i = 0; kt != php.etaMin.end(); ++kt)
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etaMax.size() << " etaMax values";
  for (kt = php.etaMax.begin(), i = 0; kt != php.etaMax.end(); ++kt)
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etaRange.size() << " etaRange values";
  for (it = php.etaRange.begin(), i = 0; it != php.etaRange.end(); ++it)
    std::cout << " [" << ++i << "] = " << (*it);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.gparHF.size() << " gparHF values";
  for (it = php.gparHF.begin(), i = 0; it != php.gparHF.end(); ++it)
    std::cout << " [" << ++i << "] = " << (*it) / CLHEP::cm;
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.noff.size() << " noff values";
  for (kt = php.noff.begin(), i = 0; kt != php.noff.end(); ++kt)
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.Layer0Wt.size() << " Layer0Wt values";
  for (it = php.Layer0Wt.begin(), i = 0; it != php.Layer0Wt.end(); ++it)
    std::cout << " [" << ++i << "] = " << (*it);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.HBGains.size() << " Shift/Gains values for HB";
  for (unsigned k = 0; k < php.HBGains.size(); ++k)
    std::cout << " [" << k << "] = " << php.HBShift[k] << ":" << php.HBGains[k];
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.HEGains.size() << " Shift/Gains values for HE";
  for (unsigned k = 0; k < php.HEGains.size(); ++k)
    std::cout << " [" << k << "] = " << php.HEShift[k] << ":" << php.HEGains[k];
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.HFGains.size() << " Shift/Gains values for HF";
  for (unsigned k = 0; k < php.HFGains.size(); ++k)
    std::cout << " [" << k << "] = " << php.HFShift[k] << ":" << php.HFGains[k];
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.etagroup.size() << " etagroup values";
  for (kt = php.etagroup.begin(), i = 0; kt != php.etagroup.end(); ++kt)
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.phigroup.size() << " phigroup values";
  for (kt = php.phigroup.begin(), i = 0; kt != php.phigroup.end(); ++kt)
    std::cout << " [" << ++i << "] = " << (*kt);
  std::cout << std::endl;
  std::cout << "HcalParametersFromDD: " << php.layerGroupEtaRec.size() << " layerGroupEtaRec blocks" << std::endl;
  for (unsigned int k = 0; k < php.layerGroupEtaRec.size(); ++k) {
    std::cout << "layerGroupEtaRec[" << k << "] Layer " << php.layerGroupEtaRec[k].layer;
    for (kt = php.layerGroupEtaRec[k].layerGroup.begin(), i = 0; kt != php.layerGroupEtaRec[k].layerGroup.end(); ++kt)
      std::cout << " " << ++i << ":" << (*kt);
    std::cout << std::endl;
  }
  std::cout << "HcalParametersFromDD: (topology|trigger)Mode " << std::hex << php.topologyMode << std::dec << std::endl;
#endif

  return true;
}

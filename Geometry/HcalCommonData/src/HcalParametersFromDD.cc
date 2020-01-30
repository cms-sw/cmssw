#include "Geometry/HcalCommonData/interface/HcalParametersFromDD.h"
#include "Geometry/HcalCommonData/interface/HcalGeomParameters.h"
#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"
#include "CondFormats/GeometryObjects/interface/HcalParameters.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
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

  int getTopologyMode(const std::string& s, bool type) {
    int result(-1);
    if (type) {
      StringToEnumParser<HcalTopologyMode::Mode> eparser;
      HcalTopologyMode::Mode mode = (HcalTopologyMode::Mode)eparser.parseString(s);
      result = static_cast<int>(mode);
    } else {
      StringToEnumParser<HcalTopologyMode::TriggerMode> eparser;
      HcalTopologyMode::TriggerMode mode = (HcalTopologyMode::TriggerMode)eparser.parseString(s);
      result = static_cast<int>(mode);
    }
    return result;
  }
}  // namespace

bool HcalParametersFromDD::build(const DDCompactView* cpv, HcalParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::build(const DDCompactView*, HcalParameters&) is called";
#endif
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
    rescale(php.rTable, HcalGeomParameters::k_ScaleFromDDDToG4);
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
    php.etaRange = DDVectorGetter::get("etaRange");
    php.gparHF = DDVectorGetter::get("gparHF");
    rescale(php.gparHF, HcalGeomParameters::k_ScaleFromDDDToG4);
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

bool HcalParametersFromDD::build(const cms::DDCompactView* cpv, HcalParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD::build(const cms::DDCompactView*, HcalParameters&) is called";
#endif
  //Special parameters at simulation level
  cms::DDFilteredView fv1(cpv->detector(), cpv->detector()->worldVolume());
  cms::DDVectorsMap vmap = cpv->detector()->vectors();
  std::string attribute = "OnlyForHcalSimNumbering";
  cms::DDSpecParRefs ref1;
  const cms::DDSpecParRegistry& mypar1 = cpv->specpars();
  mypar1.filter(ref1, attribute, "HCAL");
  fv1.mergedSpecifics(ref1);

  const int nEtaMax = 100;

  if (fv1.firstChild()) {
    std::unique_ptr<HcalGeomParameters> geom = std::make_unique<HcalGeomParameters>();
    geom->loadGeometry(cpv, php);
    php.modHB = geom->getModHalfHBHE(0);
    php.modHE = geom->getModHalfHBHE(1);
    php.dzVcal = geom->getConstDzHF();
    geom->getConstRHO(php.rHO);

    for (auto const& it : vmap) {
      if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "phioff")) {
        for (const auto& i : it.second)
          php.phioff.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "etaTable")) {
        for (const auto& i : it.second)
          php.etaTable.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "rTable")) {
        for (const auto& i : it.second)
          php.rTable.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "phibin")) {
        for (const auto& i : it.second)
          php.phibin.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "phitable")) {
        for (const auto& i : it.second)
          php.phitable.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "etaMin")) {
        for (const auto& i : it.second)
          php.etaMin.emplace_back(std::round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "etaMax")) {
        for (const auto& i : it.second)
          php.etaMax.emplace_back(std::round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "etaRange")) {
        for (const auto& i : it.second)
          php.etaRange.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "gparHF")) {
        for (const auto& i : it.second)
          php.gparHF.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "noff")) {
        for (const auto& i : it.second)
          php.noff.emplace_back(std::round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "Layer0Wt")) {
        for (const auto& i : it.second)
          php.Layer0Wt.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "HBGains")) {
        for (const auto& i : it.second)
          php.HBGains.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "HBShift")) {
        for (const auto& i : it.second)
          php.HBShift.emplace_back(round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "HEGains")) {
        for (const auto& i : it.second)
          php.HEGains.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "HEShift")) {
        for (const auto& i : it.second)
          php.HEShift.emplace_back(round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "HFGains")) {
        for (const auto& i : it.second)
          php.HFGains.emplace_back(i);
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "HFShift")) {
        for (const auto& i : it.second)
          php.HFShift.emplace_back(round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "MaxDepth")) {
        for (const auto& i : it.second)
          php.maxDepth.emplace_back(round(i));
      }
    }
    rescale(php.rTable, HcalGeomParameters::k_ScaleFromDD4HepToG4);
    rescale(php.gparHF, HcalGeomParameters::k_ScaleFromDD4HepToG4);
    for (unsigned int i = 1; i <= nEtaMax; ++i) {
      std::stringstream sstm;
      sstm << "layerGroupSimEta" << i;
      std::string tempName = sstm.str();
      for (auto const& it : vmap) {
        if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), tempName)) {
          HcalParameters::LayerItem layerGroupEta;
          layerGroupEta.layer = i;
          for (const auto& i : it.second)
            layerGroupEta.layerGroup.emplace_back(round(i));
          php.layerGroupEtaSim.emplace_back(layerGroupEta);
          break;
        }
      }
    }
  } else {
    throw cms::Exception("HcalParametersFromDD") << "Not found " << attribute.c_str() << " but needed.";
  }

  //Special parameters at reconstruction level
  cms::DDFilteredView fv2(cpv->detector(), cpv->detector()->worldVolume());
  attribute = "OnlyForHcalRecNumbering";
  cms::DDSpecParRefs ref2;
  const cms::DDSpecParRegistry& mypar2 = cpv->specpars();
  mypar2.filter(ref2, attribute, "HCAL");
  fv2.mergedSpecifics(ref2);
  if (fv2.firstChild()) {
    std::vector<std::string> tempS = fv2.get<std::vector<std::string> >("hcal", "TopologyMode");
    std::string sv = (!tempS.empty()) ? tempS[0] : "HcalTopologyMode::SLHC";
    int topoMode = getTopologyMode(sv, true);
    tempS = fv2.get<std::vector<std::string> >("hcal", "TriggerMode");
    sv = (!tempS.empty()) ? tempS[0] : "HcalTopologyMode::TriggerMode_2021";
    int trigMode = getTopologyMode(sv, false);
    php.topologyMode = ((trigMode & 0xFF) << 8) | (topoMode & 0xFF);
    for (auto const& it : vmap) {
      if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "etagroup")) {
        for (const auto& i : it.second)
          php.etagroup.emplace_back(round(i));
      } else if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), "phigroup")) {
        for (const auto& i : it.second)
          php.phigroup.emplace_back(round(i));
      }
    }
    for (unsigned int i = 1; i <= nEtaMax; ++i) {
      std::stringstream sstm;
      sstm << "layerGroupRecEta" << i;
      std::string tempName = sstm.str();
      for (auto const& it : vmap) {
        if (cms::dd::compareEqual(cms::dd::noNamespace(it.first), tempName)) {
          HcalParameters::LayerItem layerGroupEta;
          layerGroupEta.layer = i;
          for (const auto& i : it.second)
            layerGroupEta.layerGroup.emplace_back(round(i));
          php.layerGroupEtaRec.emplace_back(layerGroupEta);
          break;
        }
      }
    }
  } else {
    throw cms::Exception("HcalParametersFromDD") << "Not found " << attribute.c_str() << " but needed.";
  }

  return build(php);
}

bool HcalParametersFromDD::build(HcalParameters& php) {
  php.etaMin[0] = 1;
  if (php.etaMax[1] >= php.etaMin[1])
    php.etaMax[1] = static_cast<int>(php.etaTable.size()) - 1;
  php.etaMax[2] = php.etaMin[2] + static_cast<int>(php.rTable.size()) - 2;

  for (unsigned int i = 0; i < php.rTable.size(); ++i) {
    unsigned int k = php.rTable.size() - i - 1;
    php.etaTableHF.emplace_back(-log(tan(0.5 * atan(php.rTable[k] / php.gparHF[4]))));
  }

#ifdef EDM_ML_DEBUG
  int i(0);
  std::stringstream ss0;
  ss0 << "HcalParametersFromDD: MaxDepth[" << php.maxDepth.size() << "]: ";
  for (const auto& it : php.maxDepth)
    ss0 << it << ", ";
  edm::LogVerbatim("HCalGeom") << ss0.str();
  std::stringstream ss1;
  ss1 << "HcalParametersFromDD: ModHB [" << php.modHB.size() << "]: ";
  for (const auto& it : php.modHB)
    ss1 << it << ", ";
  edm::LogVerbatim("HCalGeom") << ss1.str();
  std::stringstream ss2;
  ss2 << "HcalParametersFromDD: ModHE [" << php.modHE.size() << "]: ";
  for (const auto& it : php.modHE)
    ss2 << it << ", ";
  edm::LogVerbatim("HCalGeom") << ss2.str();
  std::stringstream ss3;
  ss3 << "HcalParametersFromDD: " << php.phioff.size() << " phioff values:";
  std::vector<double>::const_iterator it;
  for (it = php.phioff.begin(), i = 0; it != php.phioff.end(); ++it)
    ss3 << " [" << ++i << "] = " << convertRadToDeg(*it);
  edm::LogVerbatim("HCalGeom") << ss3.str();
  std::stringstream ss4;
  ss4 << "HcalParametersFromDD: " << php.etaTable.size() << " entries for etaTable:";
  for (it = php.etaTable.begin(), i = 0; it != php.etaTable.end(); ++it)
    ss4 << " [" << ++i << "] = " << (*it);
  edm::LogVerbatim("HCalGeom") << ss4.str();
  std::stringstream ss5;
  ss5 << "HcalParametersFromDD: " << php.rTable.size() << " entries for rTable:";
  for (it = php.rTable.begin(), i = 0; it != php.rTable.end(); ++it)
    ss5 << " [" << ++i << "] = " << convertMmToCm(*it);
  edm::LogVerbatim("HCalGeom") << ss5.str();
  std::stringstream ss6;
  ss6 << "HcalParametersFromDD: " << php.phibin.size() << " phibin values:";
  for (it = php.phibin.begin(), i = 0; it != php.phibin.end(); ++it)
    ss6 << " [" << ++i << "] = " << convertRadToDeg(*it);
  edm::LogVerbatim("HCalGeom") << ss6.str();
  std::stringstream ss7;
  ss7 << "HcalParametersFromDD: " << php.phitable.size() << " phitable values:";
  for (it = php.phitable.begin(), i = 0; it != php.phitable.end(); ++it)
    ss7 << " [" << ++i << "] = " << convertRadToDeg(*it);
  edm::LogVerbatim("HCalGeom") << ss7.str();
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD: " << php.layerGroupEtaSim.size() << " layerGroupEtaSim blocks"
                               << std::endl;
  std::vector<int>::const_iterator kt;
  for (unsigned int k = 0; k < php.layerGroupEtaSim.size(); ++k) {
    std::stringstream ss8;
    ss8 << "layerGroupEtaSim[" << k << "] Layer " << php.layerGroupEtaSim[k].layer;
    for (kt = php.layerGroupEtaSim[k].layerGroup.begin(), i = 0; kt != php.layerGroupEtaSim[k].layerGroup.end(); ++kt)
      ss8 << " " << ++i << ":" << (*kt);
    edm::LogVerbatim("HCalGeom") << ss8.str();
  }
  std::stringstream ss8;
  ss8 << "HcalParametersFromDD: " << php.etaMin.size() << " etaMin values:";
  for (kt = php.etaMin.begin(), i = 0; kt != php.etaMin.end(); ++kt)
    ss8 << " [" << ++i << "] = " << (*kt);
  edm::LogVerbatim("HCalGeom") << ss8.str();
  std::stringstream ss9;
  ss9 << "HcalParametersFromDD: " << php.etaMax.size() << " etaMax values:";
  for (kt = php.etaMax.begin(), i = 0; kt != php.etaMax.end(); ++kt)
    ss9 << " [" << ++i << "] = " << (*kt);
  edm::LogVerbatim("HCalGeom") << ss9.str();
  std::stringstream ss10;
  ss10 << "HcalParametersFromDD: " << php.etaRange.size() << " etaRange values:";
  for (it = php.etaRange.begin(), i = 0; it != php.etaRange.end(); ++it)
    ss10 << " [" << ++i << "] = " << (*it);
  edm::LogVerbatim("HCalGeom") << ss10.str();
  std::stringstream ss11;
  ss11 << "HcalParametersFromDD: " << php.gparHF.size() << " gparHF values:";
  for (it = php.gparHF.begin(), i = 0; it != php.gparHF.end(); ++it)
    ss11 << " [" << ++i << "] = " << convertMmToCm(*it);
  edm::LogVerbatim("HCalGeom") << ss11.str();
  std::stringstream ss12;
  ss12 << "HcalParametersFromDD: " << php.noff.size() << " noff values:";
  for (kt = php.noff.begin(), i = 0; kt != php.noff.end(); ++kt)
    ss12 << " [" << ++i << "] = " << (*kt);
  edm::LogVerbatim("HCalGeom") << ss12.str();
  std::stringstream ss13;
  ss13 << "HcalParametersFromDD: " << php.Layer0Wt.size() << " Layer0Wt values:";
  for (it = php.Layer0Wt.begin(), i = 0; it != php.Layer0Wt.end(); ++it)
    ss13 << " [" << ++i << "] = " << (*it);
  edm::LogVerbatim("HCalGeom") << ss13.str();
  std::stringstream ss14;
  ss14 << "HcalParametersFromDD: " << php.HBGains.size() << " Shift/Gains values for HB:";
  for (unsigned k = 0; k < php.HBGains.size(); ++k)
    ss14 << " [" << k << "] = " << php.HBShift[k] << ":" << php.HBGains[k];
  edm::LogVerbatim("HCalGeom") << ss14.str();
  std::stringstream ss15;
  ss15 << "HcalParametersFromDD: " << php.HEGains.size() << " Shift/Gains values for HE:";
  for (unsigned k = 0; k < php.HEGains.size(); ++k)
    ss15 << " [" << k << "] = " << php.HEShift[k] << ":" << php.HEGains[k];
  edm::LogVerbatim("HCalGeom") << ss15.str();
  std::stringstream ss16;
  ss16 << "HcalParametersFromDD: " << php.HFGains.size() << " Shift/Gains values for HF:";
  for (unsigned k = 0; k < php.HFGains.size(); ++k)
    ss16 << " [" << k << "] = " << php.HFShift[k] << ":" << php.HFGains[k];
  edm::LogVerbatim("HCalGeom") << ss16.str();
  std::stringstream ss17;
  ss17 << "HcalParametersFromDD: " << php.etagroup.size() << " etagroup values:";
  for (kt = php.etagroup.begin(), i = 0; kt != php.etagroup.end(); ++kt)
    ss17 << " [" << ++i << "] = " << (*kt);
  edm::LogVerbatim("HCalGeom") << ss17.str();
  std::stringstream ss18;
  ss18 << "HcalParametersFromDD: " << php.phigroup.size() << " phigroup values:";
  for (kt = php.phigroup.begin(), i = 0; kt != php.phigroup.end(); ++kt)
    ss18 << " [" << ++i << "] = " << (*kt);
  edm::LogVerbatim("HCalGeom") << ss18.str();
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD: " << php.layerGroupEtaRec.size() << " layerGroupEtaRec blocks";
  for (unsigned int k = 0; k < php.layerGroupEtaRec.size(); ++k) {
    std::stringstream ss19;
    ss19 << "layerGroupEtaRec[" << k << "] Layer " << php.layerGroupEtaRec[k].layer;
    for (kt = php.layerGroupEtaRec[k].layerGroup.begin(), i = 0; kt != php.layerGroupEtaRec[k].layerGroup.end(); ++kt)
      ss19 << " " << ++i << ":" << (*kt);
    edm::LogVerbatim("HCalGeom") << ss19.str();
  }
  edm::LogVerbatim("HCalGeom") << "HcalParametersFromDD: (topology|trigger)Mode " << std::hex << php.topologyMode
                               << std::dec;
#endif

  return true;
}

void HcalParametersFromDD::rescale(std::vector<double>& v, const double s) {
  std::for_each(v.begin(), v.end(), [s](double& n) { n *= s; });
}

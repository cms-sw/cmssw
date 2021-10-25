#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB06BeamParametersFromDD.h"

bool HcalTB06BeamParametersFromDD::build(const DDCompactView* cpv,
                                         HcalTB06BeamParameters& php,
                                         const std::string& name1,
                                         const std::string& name2) {
  DDSpecificsMatchesValueFilter filter1{DDValue("Volume", name1, 0)};
  DDFilteredView fv1(*cpv, filter1);
  php.wchambers_ = getNames(fv1);

  DDSpecificsMatchesValueFilter filter2{DDValue("ReadOutName", name2, 0)};
  DDFilteredView fv2(*cpv, filter2);
  bool dodet = fv2.firstChild();
  std::vector<std::string> matNames;
  std::vector<int> nocc;
  while (dodet) {
    std::string matName = fv2.logicalPart().material().name().name();
    bool notIn = true;
    for (unsigned int i = 0; i < matNames.size(); i++) {
      if (matName == matNames[i]) {
        notIn = false;
        nocc[i]++;
      }
    }
    if (notIn) {
      matNames.push_back(matName);
      nocc.push_back(0);
    }
    dodet = fv2.next();
  }
  return build(php, matNames, nocc, name1, name2);
}

bool HcalTB06BeamParametersFromDD::build(const cms::DDCompactView* cpv,
                                         HcalTB06BeamParameters& php,
                                         const std::string& name1,
                                         const std::string& name2) {
  const cms::DDFilter filter1("Volume", name1);
  cms::DDFilteredView fv1(*cpv, filter1);
  php.wchambers_ = getNames(fv1);

  const cms::DDFilter filter2("ReadOutName", name2);
  cms::DDFilteredView fv2(*cpv, filter2);
  std::vector<std::string> matNames;
  std::vector<int> nocc;
  while (fv2.firstChild()) {
    std::string matName = static_cast<std::string>(dd4hep::dd::noNamespace(fv2.materialName()));
    ;
    bool notIn = true;
    for (unsigned int i = 0; i < matNames.size(); i++) {
      if (matName == matNames[i]) {
        notIn = false;
        nocc[i]++;
      }
    }
    if (notIn) {
      matNames.push_back(matName);
      nocc.push_back(0);
    }
  }
  return build(php, matNames, nocc, name1, name2);
}

bool HcalTB06BeamParametersFromDD::build(HcalTB06BeamParameters& php,
                                         const std::vector<std::string>& matNames,
                                         const std::vector<int>& nocc,
#ifdef EDM_ML_DEBUG
                                         const std::string& name1,
                                         const std::string& name2) {
  edm::LogVerbatim("HcalTBSim") << "HcalTB06BeamParametersFromDD:: Names to be tested for Volume = " << name1 << ": "
                                << php.wchambers_.size() << " paths";
  for (unsigned int i = 0; i < php.wchambers_.size(); i++)
    edm::LogVerbatim("HcalTBSim") << "HcalTB06BeamParametersFromDD:: (" << i << ") " << php.wchambers_[i];
#else
                                         const std::string&,
                                         const std::string&) {
#endif

  if (!matNames.empty()) {
    php.material_ = matNames[0];
    int occ = nocc[0];
    for (unsigned int i = 0; i < matNames.size(); i++) {
      if (nocc[i] > occ) {
        occ = nocc[i];
        php.material_ = matNames[i];
      }
    }
  } else {
    php.material_ = "Not Found";
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalTBSim") << "HcalTB06BeamParametersFromDD: finds " << matNames.size() << " materials for "
                                << name2;
  for (unsigned k = 0; k < matNames.size(); ++k)
    edm::LogVerbatim("HcalTBSim") << "[" << k << "] " << matNames[k] << "   " << nocc[k];
  edm::LogVerbatim("HcalTBSim") << "HcalTB06BeamParametersFromDD: Material name for ReadOut = " << name2 << ":"
                                << php.material_;
#endif
  return true;
}

std::vector<std::string> HcalTB06BeamParametersFromDD::getNames(DDFilteredView& fv) {
  std::vector<std::string> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart& log = fv.logicalPart();
    bool ok = true;
    for (unsigned int i = 0; i < tmp.size(); i++) {
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

std::vector<std::string> HcalTB06BeamParametersFromDD::getNames(cms::DDFilteredView& fv) {
  std::vector<std::string> tmp;
  while (fv.firstChild()) {
    std::string name = static_cast<std::string>(dd4hep::dd::noNamespace(fv.name()));
    if (std::find(std::begin(tmp), std::end(tmp), name) == std::end(tmp))
      tmp.emplace_back(name);
  }
  return tmp;
}

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HcalTestBeamData/interface/HcalTB02ParametersFromDD.h"

bool HcalTB02ParametersFromDD::build(const DDCompactView* cpv, HcalTB02Parameters& php, const std::string& name) {
  DDSpecificsMatchesValueFilter filter{DDValue("ReadOutName", name, 0)};
  DDFilteredView fv(*cpv, filter);
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDSolid& sol = fv.logicalPart().solid();
    const std::vector<double>& paras = sol.parameters();
    std::string namx = static_cast<std::string>(sol.name().name());
    edm::LogVerbatim("HcalTBSim") << "HcalTB02ParametersFromDD (for " << name << "): Solid " << namx << " Shape "
                                  << sol.shape() << " Parameter 0 = " << paras[0];
    if (sol.shape() == DDSolidShape::ddtrap) {
      double dz = 2 * k_ScaleFromDDDToG4 * paras[0];
      php.lengthMap_.insert(std::pair<std::string, double>(namx, dz));
    }
    dodet = fv.next();
  }
  edm::LogVerbatim("HcalTBSim") << "HcalTB02ParametersFromDD: Length Table for ReadOutName = " << name << ":";
  std::map<std::string, double>::const_iterator it = php.lengthMap_.begin();
  int i = 0;
  for (; it != php.lengthMap_.end(); it++, i++) {
    edm::LogVerbatim("HcalTBSim") << " " << i << " " << it->first << " L = " << it->second;
  }
  return true;
}

bool HcalTB02ParametersFromDD::build(const cms::DDCompactView* cpv, HcalTB02Parameters& php, const std::string& name) {
  const cms::DDFilter filter("ReadOutName", name);
  cms::DDFilteredView fv(*cpv, filter);
  while (fv.firstChild()) {
    std::string namx = static_cast<std::string>(dd4hep::dd::noNamespace(fv.name()));
    const std::vector<double>& paras = fv.parameters();
    edm::LogVerbatim("HcalTBSim") << "HcalTB02ParametersFromDD (for " << name << "): Solid " << namx << " Shape "
                                  << cms::dd::name(cms::DDSolidShapeMap, fv.shape()) << " Parameter 0 = " << paras[0];
    if (dd4hep::isA<dd4hep::Trap>(fv.solid())) {
      double dz = 2 * k_ScaleFromDD4HepToG4 * paras[0];
      php.lengthMap_.insert(std::pair<std::string, double>(namx, dz));
    }
  }
  edm::LogVerbatim("HcalTBSim") << "HcalTB02ParametersFromDD: Length Table for ReadOutName = " << name << ":";
  std::map<std::string, double>::const_iterator it = php.lengthMap_.begin();
  int i = 0;
  for (; it != php.lengthMap_.end(); it++, i++) {
    edm::LogVerbatim("HcalTBSim") << " " << i << " " << it->first << " L = " << it->second;
  }
  return true;
}

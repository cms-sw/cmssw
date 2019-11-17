#include "CondFormats/GeometryObjects/interface/EcalSimulationParameters.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "Geometry/EcalCommonData/interface/EcalSimParametersFromDD.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

using namespace geant_units::operators;

template <typename T>
void myPrint(std::string value, const std::vector<T>& vec) {
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersFromDD: " << vec.size() << " entries for " << value << ":";
  unsigned int i(0);
  for (const auto& e : vec) {
    edm::LogVerbatim("EcalGeom") << " (" << i << ") " << e;
    ++i;
  }
}

bool EcalSimParametersFromDD::build(const DDCompactView* cpv, const std::string& name, EcalSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom")
      << "Inside EcalSimParametersFromDD::build(const DDCompactView*, const std::string&, EcalSimulationParameters&)";
#endif
  // Get the filtered view
  std::string attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute, name, 0)};
  DDFilteredView fv(*cpv, filter);
  bool dodet = fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  //First the specpars
  php.useWeight_ = true;
  std::vector<double> tempD = getDDDArray("EnergyWeight", sv);
  if (!tempD.empty()) {
    if (tempD[0] < 0.1)
      php.useWeight_ = false;
  }
  tempD = getDDDArray("nxtalEta", sv);
  if (tempD.empty())
    php.nxtalEta_ = 0;
  else
    php.nxtalEta_ = static_cast<int>(tempD[0]);
  tempD = getDDDArray("nxtalPhi", sv);
  if (tempD.empty())
    php.nxtalPhi_ = 0;
  else
    php.nxtalPhi_ = static_cast<int>(tempD[0]);
  tempD = getDDDArray("PhiBaskets", sv);
  if (tempD.empty())
    php.phiBaskets_ = 0;
  else
    php.phiBaskets_ = static_cast<int>(tempD[0]);
  php.etaBaskets_ = dbl_to_int(getDDDArray("EtaBaskets", sv));
  tempD = getDDDArray("ncrys", sv);
  if (tempD.empty())
    php.ncrys_ = 0;
  else
    php.ncrys_ = static_cast<int>(tempD[0]);
  tempD = getDDDArray("nmods", sv);
  if (tempD.empty())
    php.nmods_ = 0;
  else
    php.nmods_ = static_cast<int>(tempD[0]);

  std::vector<std::string> tempS = getStringArray("Depth1Name", sv);
  if (!tempS.empty())
    php.depth1Name_ = tempS[0];
  else
    php.depth1Name_ = " ";
  tempS = getStringArray("Depth2Name", sv);
  if (!tempS.empty())
    php.depth2Name_ = tempS[0];
  else
    php.depth2Name_ = " ";

  //Then the logical volumes
  while (dodet) {
    if (std::find(php.lvNames_.begin(), php.lvNames_.end(), fv.logicalPart().name().name()) == php.lvNames_.end()) {
      php.matNames_.emplace_back(fv.logicalPart().material().name().name());
      php.lvNames_.emplace_back(fv.logicalPart().name().name());
      const DDSolid& sol = fv.logicalPart().solid();
      const std::vector<double>& paras = sol.parameters();
      double dz = (sol.shape() == DDSolidShape::ddtrap) ? (2 * paras[0]) : 0.0;
      php.dzs_.emplace_back(dz);
    }
    dodet = fv.next();
  }
  return this->buildParameters(php);
}

bool EcalSimParametersFromDD::build(const cms::DDCompactView* cpv,
                                    const std::string& name,
                                    EcalSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "Inside EcalSimParametersFromDD::build(const cms::DDCompactView*, const std::string, "
                                  "EcalSimulationParameters&)";
#endif
  // Get the filtered view
  std::string attribute = "ReadOutName";
  cms::DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());

  //First the specpars
  std::string specName = ((name == "EcalHitsEE") ? "ecal_ee" : ((name == "EcalHitsES") ? "ecal_sf" : "ecal_eb"));

  php.useWeight_ = true;
  std::vector<double> tempD = fv.get<std::vector<double> >(specName, "EnergyWeight");
  if (!tempD.empty()) {
    if (tempD[0] < 0.1)
      php.useWeight_ = false;
  }
  tempD = fv.get<std::vector<double> >(specName, "nxtalEta");
  if (tempD.empty())
    php.nxtalEta_ = 0;
  else
    php.nxtalEta_ = static_cast<int>(tempD[0]);
  tempD = fv.get<std::vector<double> >(specName, "nxtalPhi");
  if (tempD.empty())
    php.nxtalPhi_ = 0;
  else
    php.nxtalPhi_ = static_cast<int>(tempD[0]);
  tempD = fv.get<std::vector<double> >(specName, "PhiBaskets");
  if (tempD.empty())
    php.phiBaskets_ = 0;
  else
    php.phiBaskets_ = static_cast<int>(tempD[0]);
  php.etaBaskets_ = dbl_to_int(fv.get<std::vector<double> >(specName, "EtaBaskets"));
  tempD = fv.get<std::vector<double> >(specName, "ncrys");
  if (tempD.empty())
    php.ncrys_ = 0;
  else
    php.ncrys_ = static_cast<int>(tempD[0]);
  tempD = fv.get<std::vector<double> >(specName, "nmods");
  if (tempD.empty())
    php.nmods_ = 0;
  else
    php.nmods_ = static_cast<int>(tempD[0]);

  std::vector<std::string> tempS = fv.get<std::vector<std::string> >(specName, "Depth1Name");
  if (!tempS.empty())
    php.depth1Name_ = tempS[0];
  else
    php.depth1Name_ = " ";
  tempS = fv.get<std::vector<std::string> >(specName, "Depth2Name");
  if (!tempS.empty())
    php.depth2Name_ = tempS[0];
  else
    php.depth2Name_ = " ";

  //Then the logical volumes
  cms::DDSpecParRefs refs;
  const cms::DDSpecParRegistry& mypar = cpv->specpars();
  mypar.filter(refs, attribute, name);
  fv.mergedSpecifics(refs);
  while (fv.firstChild()) {
    const std::string name{fv.name().data(), fv.name().size()};
    const std::string matName{cms::dd::noNamespace(fv.materialName()).data(),
                              cms::dd::noNamespace(fv.materialName()).size()};
    if (std::find(php.lvNames_.begin(), php.lvNames_.end(), name) == php.lvNames_.end()) {
      php.matNames_.emplace_back(matName);
      php.lvNames_.emplace_back(name);
      const std::vector<double>& paras = fv.parameters();
      double dz = (fv.isATrapezoid()) ? convertCmToMm(2 * paras[0]) : 0.0;
      php.dzs_.emplace_back(dz);
    }
  };

  return this->buildParameters(php);
}

bool EcalSimParametersFromDD::buildParameters(const EcalSimulationParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersFromDD:: nxtalEta:" << php.nxtalEta_
                               << " nxtalPhi:" << php.nxtalPhi_ << " phiBaskets:" << php.phiBaskets_
                               << " ncrys:" << php.ncrys_ << " nmods: " << php.nmods_ << " useWeight:" << php.useWeight_
                               << " DeothNames:" << php.depth1Name_ << ":" << php.depth2Name_;
  myPrint("etaBaskets", php.etaBaskets_);
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersFromDD:: " << php.lvNames_.size() << " lvNames, "
                               << php.matNames_.size() << " matNames and " << php.dzs_.size() << "dzs";
#endif

  return true;
}

std::vector<double> EcalSimParametersFromDD::getDDDArray(const std::string& str, const DDsvalues_type& sv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersFromDD:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << value;
#endif
    const std::vector<double>& fvec = value.doubles();
    return fvec;
  } else {
    std::vector<double> fvec;
    return fvec;
  }
}

std::vector<std::string> EcalSimParametersFromDD::getStringArray(const std::string& str, const DDsvalues_type& sv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << "EcalSimParametersFromDD:getStringArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << value;
#endif
    const std::vector<std::string>& fvec = value.strings();
    return fvec;
  } else {
    std::vector<std::string> fvec;
    return fvec;
  }
}

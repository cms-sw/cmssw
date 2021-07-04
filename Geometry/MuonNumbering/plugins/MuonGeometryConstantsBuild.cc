#include "Geometry/MuonNumbering/plugins/MuonGeometryConstantsBuild.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

//#define EDM_ML_DEBUG

bool MuonGeometryConstantsBuild::build(const DDCompactView* cpv, MuonGeometryConstants& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom")
      << "MuonGeometryConstantsBuild;:build (const DDCompactView* cpv, MuonGeometryConstants& php)";
#endif
  std::string attribute = "OnlyForMuonNumbering";

  DDSpecificsHasNamedValueFilter filter(attribute);
  DDFilteredView fview((*cpv), filter);

  DDValue val2("level");
  const DDsvalues_type params(fview.mergedSpecifics());

  fview.firstChild();

  const DDsvalues_type mySpecs(fview.mergedSpecifics());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonGeometryConstantsBuild::mySpecs.size() = " << mySpecs.size();
#endif
  if (mySpecs.size() < 25) {
    edm::LogError("MuonGeometryConstantsBuild")
        << " MuonGeometryConstantsBuild: Missing SpecPars from DetectorDescription.";
    std::string msg =
        "MuonGeometryConstantsBuild does not have the appropriate number of SpecPars associated with the part //MUON.";
    throw cms::Exception("GeometryBuildFailure", msg);
  }

  DDsvalues_type::const_iterator bit = mySpecs.begin();
  DDsvalues_type::const_iterator eit = mySpecs.end();
  for (; bit != eit; ++bit) {
    if (bit->second.isEvaluated()) {
      php.addValue(bit->second.name(), static_cast<int>(bit->second.doubles()[0]));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "MuonGeometryConstantsBuild::adding DDConstant of " << bit->second.name() << " = "
                                   << static_cast<int>(bit->second.doubles()[0]);
#endif
    }
  }
  return true;
}

bool MuonGeometryConstantsBuild::build(const cms::DDCompactView* cpv, MuonGeometryConstants& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom")
      << "MuonGeometryConstantsBuild;:build (const cms::DDCompactView* cpv, MuonGeometryConstants& php)";
#endif
  cms::DDSpecParRegistry const& registry = cpv->specpars();
  auto it = registry.specpars.find("MuonCommonNumbering");
  if (it != end(registry.specpars)) {
    for (const auto& l : it->second.spars) {
      if (l.first == "OnlyForMuonNumbering") {
        for (const auto& k : it->second.numpars) {
          for (const auto& ik : k.second) {
            php.addValue(k.first, static_cast<int>(ik));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("MuonGeom") << "MuonGeometryConstantsBuild::adding DDConstant of " << k.first << " = "
                                         << static_cast<int>(ik);
#endif
          }
        }
      }
    }
  }

  return true;
}

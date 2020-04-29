#include "Geometry/MuonNumbering/interface/MuonDDDParametersBuild.h"
#include "Geometry/MuonNumbering/interface/MuonDDDParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

//#define EDM_ML_DEBUG

bool MuonDDDParametersBuild::build(const DDCompactView* cpv, MuonDDDParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "MuonDDDParametersBuild;:build (const DDCompactView* cpv, MuonDDDParameters& php)";
#endif
  std::string attribute = "OnlyForMuonNumbering";

  DDSpecificsHasNamedValueFilter filter(attribute);
  DDFilteredView fview((*cpv), filter);

  DDValue val2("level");
  const DDsvalues_type params(fview.mergedSpecifics());

  fview.firstChild();

  const DDsvalues_type mySpecs(fview.mergedSpecifics());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry") << "MuonDDDParametersBuild::mySpecs.size() = " << mySpecs.size();
#endif
  if (mySpecs.size() < 25) {
    edm::LogError("MuonDDDParametersBuild") << " MuonDDDParametersBuild: Missing SpecPars from DetectorDescription.";
    std::string msg =
        "MuonDDDParametersBuild does not have the appropriate number of SpecPars associated with the part //MUON.";
    throw cms::Exception("GeometryBuildFailure", msg);
  }

  DDsvalues_type::const_iterator bit = mySpecs.begin();
  DDsvalues_type::const_iterator eit = mySpecs.end();
  for (; bit != eit; ++bit) {
    if (bit->second.isEvaluated()) {
      php.addValue(bit->second.name(), int(bit->second.doubles()[0]));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("Geometry") << "MuonDDDParametersBuild::adding DDConstant of " << bit->second.name() << " = "
                                   << int(bit->second.doubles()[0]);
#endif
    }
  }
  return true;
}

bool MuonDDDParametersBuild::build(const cms::DDCompactView* cpv, MuonDDDParameters& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("Geometry")
      << "MuonDDDParametersBuild;:build (const cms::DDCompactView* cpv, MuonDDDParameters& php)";
#endif

  return false;
}

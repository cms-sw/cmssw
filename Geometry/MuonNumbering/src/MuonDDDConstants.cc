#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

//#define EDM_ML_DEBUG

MuonDDDConstants::MuonDDDConstants(const DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonDDDConstants;:MuonDDDConstants ( const DDCompactView& cpv ) constructor ";
#endif
  std::string attribute = "OnlyForMuonNumbering";

  DDSpecificsHasNamedValueFilter filter(attribute);
  DDFilteredView fview(cpv, filter);

  DDValue val2("level");
  const DDsvalues_type params(fview.mergedSpecifics());

  fview.firstChild();

  const DDsvalues_type mySpecs(fview.mergedSpecifics());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonDDDConstants::mySpecs.size() = " << mySpecs.size();
#endif
  if (mySpecs.size() < 25) {
    edm::LogError("MuonDDDConstants") << " MuonDDDConstants: Missing SpecPars from DetectorDescription.";
    std::string msg =
        "MuonDDDConstants does not have the appropriate number of SpecPars associated with the part //MUON.";
    throw cms::Exception("GeometryBuildFailure", msg);
  }

  DDsvalues_type::const_iterator bit = mySpecs.begin();
  DDsvalues_type::const_iterator eit = mySpecs.end();
  for (; bit != eit; ++bit) {
    if (bit->second.isEvaluated()) {
      this->addValue(bit->second.name(), int(bit->second.doubles()[0]));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("MuonGeom") << "MuonDDDConstants::adding DDConstant of " << bit->second.name() << " = "
                                   << int(bit->second.doubles()[0]);
#endif
    }
  }
}

int MuonDDDConstants::getValue(const std::string& name) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "about to look for ... " << name << std::endl;
#endif
  if (namesAndValues_.empty()) {
    edm::LogWarning("MuonGeom") << "MuonDDDConstants::getValue HAS NO VALUES!";
    throw cms::Exception("GeometryBuildFailure", "MuonDDDConstants does not have requested value for " + name);
  }

  std::map<std::string, int>::const_iterator findIt = namesAndValues_.find(name);

  if (findIt == namesAndValues_.end()) {
    edm::LogWarning("MuonGeom") << "MuonDDDConstants::getValue was asked for " << name << " and had NO clue!";
    throw cms::Exception("GeometryBuildFailure", "MuonDDDConstants does not have requested value for " + name);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonDDDConstants::Value for " << name << " is " << findIt->second;
#endif
  return findIt->second;
}

void MuonDDDConstants::addValue(const std::string& name, const int& value) { namesAndValues_[name] = value; }

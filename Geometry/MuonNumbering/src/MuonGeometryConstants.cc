#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//#define EDM_ML_DEBUG

int MuonGeometryConstants::getValue(const std::string& name) const {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "about to look for ... " << name << std::endl;
#endif
  if (namesAndValues_.empty())
    throw cms::Exception("GeometryBuildFailure", "MuonGeometryConstants does not have requested value for " + name);

  std::map<std::string, int>::const_iterator findIt = namesAndValues_.find(name);
  if (findIt == namesAndValues_.end())
    throw cms::Exception("GeometryBuildFailure", "MuonGeometryConstants does not have requested value for " + name);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonGeometryConstants::Value for " << name << " is " << findIt->second;
#endif
  return findIt->second;
}

void MuonGeometryConstants::addValue(const std::string& name, const int& value) { namesAndValues_[name] = value; }

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(MuonGeometryConstants);

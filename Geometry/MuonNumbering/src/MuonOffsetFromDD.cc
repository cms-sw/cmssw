#include "Geometry/MuonNumbering/interface/MuonOffsetFromDD.h"
#include "CondFormats/GeometryObjects/interface/MuonOffsetMap.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include <iostream>
#include <iomanip>

//#define EDM_ML_DEBUG

MuonOffsetFromDD::MuonOffsetFromDD(std::vector<std::string> name) : specpars_(name), nset_(name.size()) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonOffsetFromDD initialized with " << nset_ << " specpars";
#endif
}

bool MuonOffsetFromDD::build(const DDCompactView* cpv, MuonOffsetMap& php) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "Inside MuonOffsetFromDD::build(const DDCompactView*, MuonOffsetMap&)";
#endif

  // Loop over all the sets
  std::string attribute = "OnlyForMuonNumbering";
  std::string name;
  for (unsigned int k = 0; k < nset_; ++k) {
    name = "muonstep" + std::to_string(k);
    DDSpecificsMatchesValueFilter filter{DDValue(attribute, name, 0)};
    DDFilteredView fv(*cpv, filter);
    bool dodet = fv.firstChild();
    DDsvalues_type sv(fv.mergedSpecifics());
    int offset = getNumber("CopyNoOffset", sv);
    int tag = getNumber("CopyNoTag", sv);
    while (dodet) {
      name = fv.logicalPart().name().name();
      php.muonMap_[name] = std::make_pair(offset, tag);
      dodet = fv.next();
    }
  }
  return this->debugParameters(php);
}

bool MuonOffsetFromDD::build(const cms::DDCompactView* cpv, MuonOffsetMap& php) {
  edm::LogVerbatim("MuonGeom") << "Inside MuonOffsetFromDD::build(const cms::DDCompactView*, MuonOffsetMap&)";

  // Get the offsets and tags first
  int offsets[nset_], tags[nset_];
  cms::DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());
  for (unsigned int k = 0; k < nset_; ++k) {
    std::vector<int> off = fv.get<std::vector<int>>(specpars_[k], "CopyNoOffset");
    offsets[k] = (!off.empty()) ? off[0] : 0;
    std::vector<int> tag = fv.get<std::vector<int>>(specpars_[k], "CopyNoTag");
    tags[k] = (!tag.empty()) ? tag[0] : 0;
  }
  // Now loop over the detectors
  std::string attribute = "OnlyForMuonNumbering";
  std::string name;
  for (unsigned int k = 0; k < nset_; ++k) {
    name = "muonstep" + std::to_string(k);
    const cms::DDFilter filter(attribute, name);
    cms::DDFilteredView fv((*cpv), filter);
    while (fv.firstChild()) {
      name = static_cast<std::string>(fv.name());
      php.muonMap_[name] = std::make_pair(offsets[k], tags[k]);
    }
  }
  return this->debugParameters(php);
}

bool MuonOffsetFromDD::debugParameters(const MuonOffsetMap& php) {
  edm::LogVerbatim("MuonGeom") << "MuonOffsetFromDD: Finds " << php.muonMap_.size() << " entries in the map";
#ifdef EDM_ML_DEBUG
  unsigned int k(0);
  for (auto itr = php.muonMap_.begin(); itr != php.muonMap_.end(); ++itr, ++k) {
    edm::LogVerbatim("MuonGeom") << "[" << k << "] " << itr->first << ": (" << (itr->second).first << ", "
                                 << (itr->second).second << ")";
  }
#endif
  return true;
}

int MuonOffsetFromDD::getNumber(const std::string& str, const DDsvalues_type& sv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonOffsetFromDD::getNumbers called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
    const std::vector<double>& fvec = value.doubles();
    int nval = (!fvec.empty()) ? static_cast<int>(fvec[0]) : 0;
    return nval;
  } else {
    return 0;
  }
}

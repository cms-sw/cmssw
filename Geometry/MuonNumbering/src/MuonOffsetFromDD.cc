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

bool MuonOffsetFromDD::build(const DDCompactView* cpv, MuonOffsetMap& php) {
  edm::LogVerbatim("MuonGeom")
      << "Inside MuonOffsetFromDD::build(const DDCompactView*, MuonOffsetMap&)";

  // Loop over all the sets
  std::string attribute = "OnlyForMuonNumbering";
  std::string name;
  for (int k = 0; k < nset_; ++k) {
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
  return this->buildParameters(php);
}

bool MuonOffsetFromDD::build(const cms::DDCompactView* cpv, MuonOffsetMap& php) {
  edm::LogVerbatim("MuonGeom")
      << "Inside MuonOffsetFromDD::build(const cms::DDCompactView*, MuonOffsetMap&)";

  std::string specpars[nset_] = {"MuonCommonNumbering", "MuonBarrel", "MuonEndcap", "MuonBarrelWheels", "MuonBarrelStation1", "MuonBarrelStation2", "MuonBarrelStation3", "MuonBarrelStation4", "MuonBarrelSuperLayer", "MuonBarrelLayer", "MuonBarrelWire", "MuonRpcPlane1I", "MuonRpcPlane1O", "MuonRpcPlane2I", "MuonRpcPlane2O", "MuonRpcPlane3S", "MuonRpcPlane4", "MuonRpcChamberLeft", "MuonRpcChamberMiddle", "MuonRpcChamberRight", "MuonRpcEndcap1", "MuonRpcEndcap2", "MuonRpcEndcap3", "MuonRpcEndcap4", "MuonRpcEndcapSector", "MuonRpcEndcapChamberB1", "MuonRpcEndcapChamberB2", "MuonRpcEndcapChamberB3", "MuonRpcEndcapChamberC1", "MuonRpcEndcapChamberC2", "MuonRpcEndcapChamberC3", "MuonRpcEndcapChamberE1", "MuonRpcEndcapChamberE2", "MuonRpcEndcapChamberE3", "MuonRpcEndcapChamberF1", "MuonRpcEndcapChamberF2", "MuonRpcEndcapChamberF3", "MuonEndcapStation1", "MuonEndcapStation2", "MuonEndcapStation3", "MuonEndcapStation4", "MuonEndcapSubrings", "MuonEndcapSectors", "MuonEndcapLayers", "MuonEndcapRing1", "MuonEndcapRing2", "MuonEndcapRing3", "MuonEndcapRingA", "MuonGEMEndcap", "MuonGEMSector", "MuonGEMChamber"};

  // Get the offsets and tags first
  int offsets[nset_], tags[nset_];
  cms::DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());
  for (int k = 0; k < nset_; ++k) {
    std::vector<int> off = fv.get<std::vector<int>>(specpars[k], "CopyNoOffset");
    offsets[k] = (off.size() > 0) ? off[0] : 0;
    std::vector<int> tag = fv.get<std::vector<int>>(specpars[k], "CopyNoTag");
    tags[k] = (tag.size() > 0) ? tag[0] : 0;
  }
  // Now loop over the detectors
  std::string attribute = "OnlyForMuonNumbering";
  std::string name;
  for (int k = 0; k < nset_; ++k) {
    name = "muonstep" + std::to_string(k);
    const cms::DDFilter filter(attribute, name);
    cms::DDFilteredView fv((*cpv), filter);
    while (fv.firstChild()) {
      name = static_cast<std::string>(fv.name());
      php.muonMap_[name] = std::make_pair(offsets[k], tags[k]);
    }
  }
  return this->buildParameters(php);
}

bool MuonOffsetFromDD::buildParameters(const MuonOffsetMap& php) {
  edm::LogVerbatim("MuonGeom") << "MuonOffsetFromDD: Finds " << php.muonMap_.size() << " entries in the map";
#ifdef EDM_ML_DEBUG
  unsigned int k(0);
  for (auto itr = php.muonMap_.begin(); itr != php.muonMap_.end(); ++itr, ++k) {
    edm::LogVerbatim("MuonGeom") << "[" << k << "] " << itr->first << ": (" << (itr->second).first << ", " << (itr->second).second << ")";
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
    int nval = (fvec.size() > 0) ? static_cast<int>(fvec[0]) : 0;
    return nval;
  } else {
    return 0;
  }
}

#include "Geometry/MuonNumbering/interface/MuonGeometryNumbering.h"
#include "Geometry/MuonNumbering/interface/MuonBaseNumber.h"
#include "Geometry/MuonNumbering/interface/MuonGeometryConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

MuonGeometryNumbering::MuonGeometryNumbering(const MuonGeometryConstants &muonConstants) {
  //  Get constant values from muonConstants
  theLevelPart = muonConstants.getValue("level");
  theSuperPart = muonConstants.getValue("super");
  theBasePart = muonConstants.getValue("base");
  theStartCopyNo = muonConstants.getValue("xml_starts_with_copyno");

  // some consistency checks

  if (theBasePart != 1) {
    edm::LogWarning("MuonGeom") << "MuonGeometryNumbering finds unusual base constant:" << theBasePart;
  }
  if (theSuperPart < 100) {
    edm::LogWarning("MuonGeom") << "MuonGeometryNumbering finds unusual super constant:" << theSuperPart;
  }
  if (theLevelPart < 10 * theSuperPart) {
    edm::LogWarning("MuonGeom") << "MuonGeometryNumbering finds unusual level constant:" << theLevelPart;
  }
  if ((theStartCopyNo != 0) && (theStartCopyNo != 1)) {
    edm::LogWarning("MuonGeom") << "MuonGeometryNumbering finds unusual start value for copy numbers:"
                                << theStartCopyNo;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonGeometryNumbering configured with"
                               << " Level = " << theLevelPart << " Super = " << theSuperPart
                               << " Base = " << theBasePart << " StartCopyNo = " << theStartCopyNo;
#endif
}

MuonBaseNumber MuonGeometryNumbering::geoHistoryToBaseNumber(const DDGeoHistory &history) const {
  MuonBaseNumber num;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonGeometryNumbering create MuonBaseNumber for " << history;
#endif

  //loop over all parents and check
  DDGeoHistory::const_iterator cur = history.begin();
  DDGeoHistory::const_iterator end = history.end();
  while (cur != end) {
    const DDLogicalPart &ddlp = cur->logicalPart();
    const int tag = getInt("CopyNoTag", ddlp) / theLevelPart;
    if (tag > 0) {
      const int offset = getInt("CopyNoOffset", ddlp);
      const int copyno = (cur->copyno()) + offset % theSuperPart;
      const int super = offset / theSuperPart;
      num.addBase(tag, super, copyno - theStartCopyNo);
    }
    cur++;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MuonGeom") << "MuonGeometryNumbering::" << num.getLevels();
  for (int i = 1; i <= num.getLevels(); i++) {
    edm::LogVerbatim("MuonGeom") << "[" << i << "] " << num.getSuperNo(i) << " " << num.getBaseNo(i);
  }
#endif

  return num;
}

MuonBaseNumber MuonGeometryNumbering::geoHistoryToBaseNumber(const cms::ExpandedNodes &nodes) const {
  MuonBaseNumber num;

  int ctr(0);
  for (auto const &it : nodes.tags) {
    int tag = it / theLevelPart;
    if (tag > 0) {
      int offset = nodes.offsets[ctr];
      int copyno = nodes.copyNos[ctr] + offset % theSuperPart;
      int super = offset / theSuperPart;
      num.addBase(tag, super, copyno - theStartCopyNo);
    }
    ++ctr;
  }
  return num;
}

int MuonGeometryNumbering::getInt(const std::string &s, const DDLogicalPart &part) const {
  DDValue val(s);
  std::vector<const DDsvalues_type *> result = part.specifics();
  std::vector<const DDsvalues_type *>::iterator it = result.begin();
  bool foundIt = false;
  for (; it != result.end(); ++it) {
    foundIt = DDfetch(*it, val);
    if (foundIt)
      break;
  }
  if (foundIt) {
    std::vector<double> temp = val.doubles();
    if (temp.size() != 1) {
      edm::LogError("MuonGeom") << "MuonGeometryNumbering:: ERROR: I need only 1 " << s << " in DDLogicalPart "
                                << part.name();
      abort();
    }
    return int(temp[0]);
  } else
    return 0;
}

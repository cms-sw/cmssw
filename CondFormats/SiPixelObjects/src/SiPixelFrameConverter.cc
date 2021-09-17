#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace std;
using namespace sipixelobjects;

SiPixelFrameConverter::SiPixelFrameConverter(const SiPixelFedCabling* map, int fedId)
    : theFedId(fedId),
      theMap(map),
      theTree(dynamic_cast<SiPixelFedCablingTree const*>(map)),
      theFed(theTree ? theTree->fed(fedId) : nullptr) {}

bool SiPixelFrameConverter::hasDetUnit(uint32_t rawId) const {
  return theMap->pathToDetUnitHasDetUnit(rawId, static_cast<unsigned int>(theFedId));
}

PixelROC const* SiPixelFrameConverter::toRoc(int link, int roc) const {
  CablingPathToDetUnit path = {
      static_cast<unsigned int>(theFedId), static_cast<unsigned int>(link), static_cast<unsigned int>(roc)};
  const PixelROC* rocp = (theFed) ? theTree->findItemInFed(path, theFed) : theMap->findItem(path);
  if UNLIKELY (!rocp) {
    stringstream stm;
    stm << "Map shows no fed=" << theFedId << ", link=" << link << ", roc=" << roc;
    edm::LogWarning("SiPixelFrameConverter") << stm.str();
  }
  return rocp;
}

int SiPixelFrameConverter::toCabling(ElectronicIndex& cabling, const DetectorIndex& detector) const {
  std::vector<CablingPathToDetUnit> path = theMap->pathToDetUnit(detector.rawId);
  typedef std::vector<CablingPathToDetUnit>::const_iterator IT;
  for (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC* roc = theMap->findItem(*it);
    if (!roc)
      return 2;
    if (roc->rawId() != detector.rawId)
      return 3;

    GlobalPixel global = {detector.row, detector.col};
    //LogTrace("")<<"GLOBAL PIXEL: row=" << global.row <<" col="<< global.col;

    LocalPixel local = roc->toLocal(global);
    // LogTrace("")<<"LOCAL PIXEL: dcol ="
    //<<  local.dcol()<<" pxid="<<  local.pxid()<<" inside: " <<local.valid();

    if (!local.valid())
      continue;
    ElectronicIndex cabIdx = {static_cast<int>(it->link), static_cast<int>(it->roc), local.dcol(), local.pxid()};
    cabling = cabIdx;
    return 0;
  }
  return 1;
}

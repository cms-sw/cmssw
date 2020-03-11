// SiPixelQuality.cc
//
// class implementation to hold a list of disabled pixel modules
//
// M. Eads
// Apr 2008

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameReverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <algorithm>
#include <iostream>

using namespace sipixelobjects;

//////////////////////////////////////
//  errortype "whole" = int 0 in DB //
//  errortype "tbmA" = int 1 in DB  //
//  errortype "tbmB" = int 2 in DB  //
//  errortype "none" = int 3 in DB  //
//////////////////////////////////////

/////////////////////////////////////////////////
//each bad roc correspond to a bit to 1: num=  //
// 0 <-> all good rocs                         //
// 1 <-> only roc 0 bad                        //
// 2<-> only roc 1 bad                         //
// 3<->  roc 0 and 1 bad                       //
// 4 <-> only roc 2 bad                        //
//  ...                                        //
/////////////////////////////////////////////////

// add a list of modules to the vector of disabled modules
void SiPixelQuality::addDisabledModule(std::vector<SiPixelQuality::disabledModuleType>& idVector) {
  theDisabledModules.insert(theDisabledModules.end(), idVector.begin(), idVector.end());
}

void SiPixelQuality::add(const SiStripDetVOff* Voff) {
  //Get vector of Voff dets
  std::vector<uint32_t> vdets;
  Voff->getDetIds(vdets);

  std::vector<uint32_t>::const_iterator iter = vdets.begin();
  std::vector<uint32_t>::const_iterator iterEnd = vdets.end();

  for (; iter != iterEnd; ++iter) {
    SiPixelQuality::disabledModuleType BadModule;
    BadModule.DetID = *iter;
    BadModule.errorType = 0;
    BadModule.BadRocs = 65535;  //NOTE: here all module is bad - no difference for half modules
    if (IsModuleUsable(BadModule.DetID))
      addDisabledModule(BadModule);
  }
}

// bool SiPixelQuality::removeDisabledModule(const uint32_t & detid){
//   std::vector<disabledModuleType>::const_iterator iter = std::lower_bound(theDisabledModules.begin(),theDisabledModules.end(),detid,SiPixelQuality::BadComponentStrictWeakOrdering());
//   if (iter != theDisabledModules.end() && iter->DetID==detid){
//     theDisabledModules.erase(iter);
//     return true;
//   }
//   return false;
// }

// ask if module is OK
bool SiPixelQuality::IsModuleUsable(const uint32_t& detid) const {
  if (IsFedBad(detid))
    return true;
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = theDisabledModules;
  std::sort(disabledModules.begin(), disabledModules.end(), SiPixelQuality::BadComponentStrictWeakOrdering());
  std::vector<disabledModuleType>::const_iterator iter = std::lower_bound(
      disabledModules.begin(), disabledModules.end(), detid, SiPixelQuality::BadComponentStrictWeakOrdering());
  if (iter != disabledModules.end() && iter->DetID == detid && iter->errorType == 0)
    return false;
  return true;
}

int SiPixelQuality::BadModuleNumber() {
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = theDisabledModules;
  //does not distinguish between partially dead or not
  return disabledModules.size();
}

//ask if module is bad
bool SiPixelQuality::IsModuleBad(const uint32_t& detid) const {
  if (IsFedBad(detid))
    return true;
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = theDisabledModules;
  std::sort(disabledModules.begin(), disabledModules.end(), SiPixelQuality::BadComponentStrictWeakOrdering());
  std::vector<disabledModuleType>::const_iterator iter = std::lower_bound(
      disabledModules.begin(), disabledModules.end(), detid, SiPixelQuality::BadComponentStrictWeakOrdering());
  //errorType 0 corresponds to "whole" dead module
  if (iter != disabledModules.end() && iter->DetID == detid && iter->errorType == 0)
    return true;
  return false;
}

//ask if roc is bad
bool SiPixelQuality::IsRocBad(const uint32_t& detid, const short& rocNb) const {
  if (IsModuleBad(detid))
    return true;
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = theDisabledModules;
  std::sort(disabledModules.begin(), disabledModules.end(), SiPixelQuality::BadComponentStrictWeakOrdering());
  std::vector<disabledModuleType>::const_iterator iter = std::lower_bound(
      disabledModules.begin(), disabledModules.end(), detid, SiPixelQuality::BadComponentStrictWeakOrdering());
  if (iter != disabledModules.end() && iter->DetID == detid) {
    return ((iter->BadRocs >> rocNb) & 0x1);
  }
  return false;
}

//ask for bad ROCS

short SiPixelQuality::getBadRocs(const uint32_t& detid) const {
  std::vector<SiPixelQuality::disabledModuleType> disabledModules = theDisabledModules;
  std::sort(disabledModules.begin(), disabledModules.end(), SiPixelQuality::BadComponentStrictWeakOrdering());
  std::vector<disabledModuleType>::const_iterator iter = std::lower_bound(
      disabledModules.begin(), disabledModules.end(), detid, SiPixelQuality::BadComponentStrictWeakOrdering());
  if (iter != disabledModules.end() && iter->DetID == detid)
    return iter->BadRocs;
  return 0;
}

const std::vector<LocalPoint> SiPixelQuality::getBadRocPositions(const uint32_t& detid,
                                                                 const TrackerGeometry& theTracker,
                                                                 const SiPixelFedCabling* map) const {
  std::vector<LocalPoint> badrocpositions(0);
  for (unsigned int i = 0; i < 16; i++) {
    if (IsRocBad(detid, i) == true) {
      std::vector<CablingPathToDetUnit> path = map->pathToDetUnit(detid);
      typedef std::vector<CablingPathToDetUnit>::const_iterator IT;
      for (IT it = path.begin(); it != path.end(); ++it) {
        const PixelROC* myroc = map->findItem(*it);
        if (myroc->idInDetUnit() == i) {
          LocalPixel::RocRowCol local = {39, 25};  //corresponding to center of ROC row, col
          GlobalPixel global = myroc->toGlobal(LocalPixel(local));
          //       edm::ESHandle<TrackerGeometry> geom;
          //     es.get<TrackerDigiGeometryRecord>().get( geom );
          //    const TrackerGeometry& theTracker(*geom);
          const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detid));

          PixelTopology const* topology = &(theGeomDet->specificTopology());

          MeasurementPoint thepoint(global.row, global.col);
          LocalPoint localpoint = topology->localPosition(thepoint);
          badrocpositions.push_back(localpoint);
          break;
        }
      }
    }
  }
  return badrocpositions;
}

bool SiPixelQuality::IsFedBad(const uint32_t& detid) const { return false; }

#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace std;
using namespace sipixelobjects;

SiPixelFrameConverter::SiPixelFrameConverter(const SiPixelFedCabling* map, int fedId)
  : theFedId(fedId), theMap(map)
{ }


bool SiPixelFrameConverter::hasDetUnit(uint32_t rawId) const
{
  std::vector<CablingPathToDetUnit> paths = theMap->pathToDetUnit(rawId);
  typedef std::vector<CablingPathToDetUnit>::const_iterator IT;
  for (IT it=paths.begin(); it!=paths.end();++it) {
    if(it->fed==static_cast<unsigned int>(theFedId)) return true;
  }
  return false;
}


int SiPixelFrameConverter::toDetector(const ElectronicIndex & cabling, DetectorIndex & detector) const
{
  CablingPathToDetUnit path = {static_cast<unsigned int>(theFedId),
                               static_cast<unsigned int>(cabling.link),
                               static_cast<unsigned int>(cabling.roc)}; 
  const PixelROC * roc = theMap->findItem(path);
  if (!roc){
    stringstream stm;
    stm << "Map shows no fed="<<theFedId
        <<", link="<<cabling.link
        <<", roc="<<cabling.roc;
    edm::LogWarning("SiPixelFrameConverter") << stm.str();
    return 2;
  }
  LocalPixel::DcolPxid local = { cabling.dcol, cabling.pxid };
  if (!local.valid()) return 3;

  GlobalPixel global = roc->toGlobal( LocalPixel(local) );
  detector.rawId = roc->rawId();
  detector.row   = global.row;
  detector.col   = global.col;

  return 0;


}


int SiPixelFrameConverter::toCabling(
    ElectronicIndex & cabling, const DetectorIndex & detector) const 
{
  std::vector<CablingPathToDetUnit> path = theMap->pathToDetUnit(detector.rawId);
  typedef  std::vector<CablingPathToDetUnit>::const_iterator IT;
  for  (IT it = path.begin(); it != path.end(); ++it) {
    const PixelROC * roc = theMap->findItem(*it); 
    if (!roc) return 2;
    if (! roc->rawId() == detector.rawId) return 3;

    GlobalPixel global = {detector.row, detector.col};
    //LogTrace("")<<"GLOBAL PIXEL: row=" << global.row <<" col="<< global.col;

    LocalPixel local = roc->toLocal(global);
    // LogTrace("")<<"LOCAL PIXEL: dcol =" 
    //<<  local.dcol()<<" pxid="<<  local.pxid()<<" inside: " <<local.valid();

    if(!local.valid()) continue;
    ElectronicIndex cabIdx = {static_cast<int>(it->link), 
                              static_cast<int>(it->roc), local.dcol(), local.pxid()}; 
    cabling = cabIdx;
    return 0;
  }  
  return 1;
}


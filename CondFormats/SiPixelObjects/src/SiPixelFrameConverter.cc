#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"

#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>

using namespace std;
using namespace sipixelobjects;

SiPixelFrameConverter::SiPixelFrameConverter(const SiPixelFedCablingMap * map, int fedId)
  : theFed( *(*map).fed(fedId))
{ }


bool SiPixelFrameConverter::hasDetUnit(uint32_t rawId) const
{
  for (int idxLink = 1; idxLink <= theFed.numberOfLinks(); idxLink++) {
    const PixelFEDLink * link = theFed.link(idxLink);
    if (!link) continue;
    int numberOfRocs = link->numberOfROCs();
    for(int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      if (!roc) continue;
      if (rawId == roc->rawId() ) return true;
    }
  }
  return false;
}


int SiPixelFrameConverter::toDetector(const ElectronicIndex & cabling, DetectorIndex & detector) const
{
  const PixelFEDLink * link = theFed.link( cabling.link);
  if (!link) {
    stringstream stm;
    stm << "FED shows no link of id= " << cabling.link;
    edm::LogError("SiPixelFrameConverter") << stm.str();
    return 1;
  }

  const PixelROC * roc = link->roc(cabling.roc);
  if (!roc) {
    stringstream stm;
    stm << "Link=" <<  cabling.link << " shows no ROC with id=" << cabling.roc;
    edm::LogError("SiPixelFrameConverter") << stm.str();
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


int SiPixelFrameConverter::toCabling(ElectronicIndex & cabling, const DetectorIndex & detector) const
{
  for (int idxLink = 1; idxLink <= theFed.numberOfLinks(); idxLink++) {
    const PixelFEDLink * link = theFed.link(idxLink);
    int linkid = link->id();
    int numberOfRocs = link->numberOfROCs();

    for(int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      if (detector.rawId == roc->rawId() ) {
        GlobalPixel global = {detector.row, detector.col};
//LogTrace("")<<"GLOBAL PIXEL: row=" << global.row <<" col="<< global.col;
        LocalPixel local = roc->toLocal(global);
// LogTrace("")<<"LOCAL PIXEL: dcol =" <<  local.dcol()<<" pxid="<<  local.pxid()<<" inside: " <<local.valid();
        if(!local.valid()) continue; 
        ElectronicIndex cabIdx = {linkid, idxRoc, local.dcol(), local.pxid()};
        cabling = cabIdx;
        return 0;
      }
    }
  }
  // proper unit not found, thrown exception
  return 1;
}


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
  for (int idxLink = 0; idxLink < theFed.numberOfLinks(); idxLink++) {
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


int SiPixelFrameConverter::toDetector(const CablingIndex & cabling, DetectorIndex & detector) const
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

  uint32_t detId = roc->rawId();
  PixelROC::LocalPixel local = {cabling.dcol, cabling.pxid};
  PixelROC::GlobalPixel global = roc->toGlobal(local); 

  DetectorIndex detIdx = {detId,  global.row, global.col};
  detector = detIdx;
  return 0;
}


int SiPixelFrameConverter::toCabling(CablingIndex & cabling, const DetectorIndex & detector) const
{
  for (int idxLink = 0; idxLink < theFed.numberOfLinks(); idxLink++) {
    const PixelFEDLink * link = theFed.link(idxLink);
    int linkid = link->id();
    int numberOfRocs = link->numberOfROCs();

    for(int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      if (detector.rawId == roc->rawId() ) {
        PixelROC::GlobalPixel global = {detector.row, detector.col};
        PixelROC::LocalPixel local = roc->toLocal(global);
        if(! roc->inside(local)) continue; 
        CablingIndex cabIdx = {linkid, idxRoc, local.dcol, local.pxid};
        cabling = cabIdx;
        return 0;
      }
    }
  }
  // proper unit not found, thrown exception
  return 1;
}


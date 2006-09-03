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
    int numberOfRocs = link->numberOfROCs();
    for(int idxRoc = 0; idxRoc < numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      if (rawId == roc->rawId()) return true;
    }
  }
  return false;
}

SiPixelFrameConverter::DetectorIndex SiPixelFrameConverter::
    toDetector(const CablingIndex & cabling) const
{
  const PixelFEDLink * link = theFed.link( cabling.link);
  if (!link) {
    stringstream stm;
    stm << "FED shows no link of id= " << cabling.link;
    throw cms::Exception(stm.str());
  }
  const PixelROC * roc = link->roc(cabling.roc);
  if (!roc) {
    stringstream stm;
    stm << "Link=" <<  cabling.link << " shows no ROC with id=" << cabling.roc;
    throw cms::Exception(stm.str());
  }

  LocalPixel local = {cabling.dcol, cabling.pxid};
  GlobalPixel global = toGlobal(*roc, local);

  DetectorIndex detIdx = {roc->rawId(),  global.row, global.col}; 
  return detIdx;
}

SiPixelFrameConverter::GlobalPixel SiPixelFrameConverter::
  toGlobal(const PixelROC &roc, const LocalPixel & loc) const 
{
    int icol, irow;
    if (loc.pxid < PixelROC::rows()) {
      icol = 0;
      irow = loc.pxid;
    }
    else {
      icol = 1;
      irow = 2*PixelROC::rows() - loc.pxid-1;
    }
    GlobalPixel res;
    res.row = PixelROC::rows()*roc.y() + irow;
    res.col = PixelROC::cols()*roc.x() + loc.dcol * 2 + icol;
    return res;
}

SiPixelFrameConverter::CablingIndex SiPixelFrameConverter::
    toCabling(const DetectorIndex & detector) const
{
  for (int idxLink = 0; idxLink < theFed.numberOfLinks(); idxLink++) {
    const PixelFEDLink * link = theFed.link(idxLink);
    int linkid = link->id();
    int numberOfRocs = link->numberOfROCs();
    for(int idxRoc = 0; idxRoc < numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      if (detector.rawId == roc->rawId()) {
        GlobalPixel global = {detector.row, detector.col};
        LocalPixel local = toLocal(*roc, global);
        if(! (*roc).inside(local.dcol, local.pxid) ) continue;
        CablingIndex cabIdx = {linkid, idxRoc, local.dcol, local.pxid};
        return cabIdx;
      }
    }
  }
  // proper unit not found, thrown exception
  throw cms::Exception("SiPixelFrameConverter::toCabling, noCabling index found!");

}

SiPixelFrameConverter::LocalPixel SiPixelFrameConverter::
    toLocal(const PixelROC &roc,  const GlobalPixel& glo) const
{
  LocalPixel loc;
  int rowRoc = glo.row / roc.rows();
  int colRoc = glo.col / roc.cols();

  if (rowRoc != roc.y() || colRoc != roc.x() ) {
    loc.dcol = -1;
    loc.pxid = -1;
    return loc;
  }

  int inRow = glo.row % roc.rows();
  int inCol = glo.col %  roc.cols();
  int icol = inCol % 2;

  loc.dcol = inCol / 2;
  loc.pxid = (icol == 0) ? inRow : 2*roc.rows() - inRow - 1;

  return loc;
}

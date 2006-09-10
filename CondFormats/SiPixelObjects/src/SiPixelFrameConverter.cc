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
      if (rawId == link->rocDetUnit(idxRoc).first ) return true;
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

  uint32_t detId = roc->rawId();
  PixelROC::LocalPixel local = {cabling.dcol, cabling.pxid};
  PixelROC::GlobalPixel global = roc->toGlobal(local); 

  PixelROC::LocalPixel aLocal = roc->toLocal(global);
  if (local.pxid != aLocal.pxid || local.dcol != aLocal.dcol) {
    ostringstream out;
    out<<"PROBLEM with inversion: Local1: "<<local.pxid<<","<<local.dcol
                              <<" Local2: "<<aLocal.pxid<<","<<aLocal.dcol;
    LogDebug("** HERE**")<<out.str();
  }

/*
  pair<uint32_t,ModuleType> detSpec = link->rocDetUnit(cabling.roc);
  const uint32_t & detId = detSpec.first;
//  pair<int,int> result = roc->convert(cabling.dcol, cabling.pxid);
  const ModuleType & detType = detSpec.second;
  
  
  pair<int,int> topolygy = rowsAndCollumns(detType);
  const int & nRows = topolygy.first;
  const int & nCols = topolygy.second;

  GlobalPixel global = toGlobal(*roc, local);
  convertModuleToDetUnit(detType,nRows,nCols, global);
  reverseDetUnitFrame(detId,nRows,nCols, global);
//  uint32_t detId = link->rocDetUnit2(cabling.roc);
  GlobalPixel global = {0,0};

*/

  DetectorIndex detIdx = {detId,  global.row, global.col}; 
  // CablingIndex c = toCabling(detIdx);
  return detIdx;
}

SiPixelFrameConverter::GlobalPixel SiPixelFrameConverter::
  toGlobal( const PixelROC &roc, const LocalPixel & loc) const 
{
    //
    // rows and collumns in ROC frame
    //
    int rocCol, rocRow;
    if (loc.pxid < PixelROC::rows()) {
      rocCol = 2*loc.dcol;
      rocRow = loc.pxid;
    }
    else {
      rocRow = 2*PixelROC::rows() - loc.pxid-1;
      rocCol = 2*loc.dcol+1;
    }

    //
    // rows and collumns in Module frame defined by first ROC frame
    //
    if ( roc.y() % 2 == 1) {
      rocRow = PixelROC::rows() - rocRow -1;
      rocCol = PixelROC::cols() - rocCol -1;
    }
    rocRow += PixelROC::rows()*roc.x();
    rocCol += PixelROC::cols()*roc.y();

    //
    // move to DetUnit frame
    //
    GlobalPixel result = {rocRow,rocCol};
    return result;
}


SiPixelFrameConverter::CablingIndex SiPixelFrameConverter::
    toCabling(const DetectorIndex & detector) const
{
  for (int idxLink = 0; idxLink < theFed.numberOfLinks(); idxLink++) {
    const PixelFEDLink * link = theFed.link(idxLink);
    int linkid = link->id();
    int numberOfRocs = link->numberOfROCs();
    LogDebug("************************************************* HERE**");
    for(int idxRoc = 0; idxRoc < numberOfRocs; idxRoc++) {
      const PixelROC * roc = link->roc(idxRoc);
      if (detector.rawId == roc->rawId() ) {
        PixelROC::GlobalPixel global = {detector.row, detector.col};
        PixelROC::LocalPixel local = roc->toLocal(global);
        if(! roc->inside(local)) continue;
        CablingIndex cabIdx = {linkid, idxRoc, local.dcol, local.pxid};
        PixelROC::GlobalPixel global2 = roc->toGlobal(local);

        if (global.row != global2.row || global.col != global2.col) {
         ostringstream out;
        out<<"PROBLEM with inversion:"
          <<" Global1: "<<global.row<<","<<global.col
          <<" Local:   "<<local.pxid<<","<<local.dcol
          <<" Global2: "<<global2.row<<","<<global2.col<<std::endl;
         LogDebug("** HERE**")<<out.str();

        }
        return cabIdx;
/*
      pair<uint32_t,ModuleType> detSpec = link->rocDetUnit(idxRoc);
      const uint32_t & detId = detSpec.first;
      if (detector.rawId == detId) {

        const ModuleType & detType = detSpec.second;
        pair<int,int> topolygy = rowsAndCollumns(detType);
        const int & nRows = topolygy.first;
        const int & nCols = topolygy.second;

        GlobalPixel global = {detector.row, detector.col};
        reverseDetUnitFrame(detId,nRows,nCols, global);
        convertModuleToDetUnit(detType,nRows,nCols, global);

        LocalPixel local = toLocal(*roc, global);
        if(! (*roc).inside(local.dcol, local.pxid) ) continue;
        CablingIndex cabIdx = {linkid, idxRoc, local.dcol, local.pxid};

        cout <<"Detector: " << "det: "<<detector.rawId
                            <<" row: "<<detector.row
                            <<" col: "<<detector.col
                            <<endl;
        DetectorIndex detector2 = toDetector(cabIdx);
        cout <<"Detectr2: " << "det: "<<detector2.rawId
                            <<" row: "<<detector2.row
                            <<" col: "<<detector2.col<<endl;
        return cabIdx;
*/
      }
    }
  }
  // proper unit not found, thrown exception
  throw cms::Exception("SiPixelFrameConverter::toCabling, noCabling index found!");

}

SiPixelFrameConverter::LocalPixel SiPixelFrameConverter::
    toLocal( const PixelROC &roc,  const GlobalPixel& glo) const
{
  LocalPixel loc;

  //
  // check that ROC is correct
  //
  int rocInX = glo.row / roc.rows();
  int rocInY = glo.col / roc.cols();
  if (rocInX != roc.x() || rocInY != roc.y() ) {
    loc.dcol = -1;
    loc.pxid = -1;
    return loc;
  }

  //
  // position in frame given by RO
  //
  int rocRow = glo.row % roc.rows();
  int rocCol = glo.col %  roc.cols();
  if (rocInY %2 ==1) {
     rocRow = PixelROC::rows() - rocRow -1;
     rocCol = PixelROC::cols() - rocCol -1;
  }

  loc.dcol = rocCol / 2;
  loc.pxid = (rocCol % 2 == 0) ? rocRow : 2*roc.rows() - rocRow - 1;

  return loc;
}


std::pair<int,int> SiPixelFrameConverter::rowsAndCollumns(const sipixelobjects::ModuleType & t) const
{
  int nRows, nColls;
  switch (t) {
    case(v1x2) : {nRows = 1*PixelROC::rows(); nColls = 2*PixelROC::cols(); break;}
    case(v1x5) : {nRows = 1*PixelROC::rows(); nColls = 5*PixelROC::cols(); break;}
    case(v1x8) : {nRows = 1*PixelROC::rows(); nColls = 8*PixelROC::cols(); break;}
    case(v2x3) : {nRows = 2*PixelROC::rows(); nColls = 3*PixelROC::cols(); break;}
    case(v2x4) : {nRows = 2*PixelROC::rows(); nColls = 4*PixelROC::cols(); break;}
    case(v2x5) : {nRows = 2*PixelROC::rows(); nColls = 5*PixelROC::cols(); break;}
    case(v2x8) : {nRows = 2*PixelROC::rows(); nColls = 8*PixelROC::cols(); break;}
    default: { nRows=0; nColls=0; edm::LogError("SiPixelFrameConverter::RowsAndCollumns, problem");}
  };
  return make_pair(nRows, nColls);
}


void SiPixelFrameConverter::convertModuleToDetUnit(const sipixelobjects::ModuleType & type, 
       int nRows, int nCols, GlobalPixel & result) const
{
    switch(type) {
      case(v1x2): case(v1x5): {
        result.col = nCols - result.col;
        break;
      }
      case(v1x8): case(v2x8): case(v2x3): case(v2x4): case(v2x5):  {
        result.row = nRows - result.row;
        break;
      }
      default: edm::LogError("SiPixelFrameConverter::toGlobal, problem");
    }
}


void SiPixelFrameConverter::reverseDetUnitFrame(
    const uint32_t & detId, int nRows, int nCols, GlobalPixel & global) const
{
  bool barrel ( 1==((detId>>25)&0x7));
  bool negative_Z ( 5 > ((detId>>2)&0xF) ); // test for barrel only
  if (barrel && negative_Z) {
     global.row = nRows - global.row;
     global.col = nCols - global.col;
  }
}

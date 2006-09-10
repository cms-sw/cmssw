#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <sstream>
using namespace std;
using namespace sipixelobjects;

int PixelROC::theNRows = 80;
int PixelROC::theNCols = 52;

PixelROC::PixelROC(uint32_t du, int idDU, int idLk, int rocInX, int rocInY)
  : 
    theDetUnit(du), 
    theIdDU(idDU), theIdLk(idLk),
    theRocInX(rocInX), theRocInY(rocInY)
{ 
  theRowOffset = 0;
  theRowSlopeSign = 1;
  theColOffset = 0;
  theColSlopeSign = 1;
}

bool PixelROC::inside( int dcol, int pxid) const 
{
  return (     0 <= dcol && dcol < theNCols/2
           &&  0 <= pxid && pxid < 2*theNRows );
}

bool PixelROC::inside( const LocalPixel & lp) const
{
  return (     0 <= lp.dcol && lp.dcol < 26
           &&  0 <= lp.pxid && lp.pxid < 160 );
}

PixelROC::GlobalPixel PixelROC:: toGlobal(const LocalPixel & loc) const 
{
  int rocCol, rocRow;
  if (loc.pxid < theNRows) {
    rocCol = loc.dcol*2;
    rocRow = loc.pxid;
  }
  else {
    rocCol = loc.dcol*2 + 1;
    rocRow = 2*theNRows - loc.pxid-1;
  }


  FrameConversion conversion(theDetUnit, theIdDU);
//    FrameConversion conversion(theRowOffset,theRowSlopeSign,theColOffset,theColSlopeSign);
  GlobalPixel result;
  result.col  = conversion.collumn().convert(rocCol);
  result.row  = conversion.row().convert(rocRow);

  {
    ostringstream out;
    out<<" detunit: "<<theDetUnit<<" ROC_id: "<<theIdDU;
    out<<" initial local pxid: "<<loc.pxid<<" dcol:"<<loc.dcol;
    out<<" intermediate local (in ROC frame) rocRow: "<<rocRow<<", rocCol: "<<rocCol;
    out<<" final global row: "<<result.row <<" colomn: "<<result.col;
    LogDebug("** HERE, toGlobal**")<<out.str();
  }
/*
  int col_inv = conversion.collumn().inverse(result.col);
  int row_inv = conversion.row().inverse(result.row);

  if ( rocCol != col_inv || row_inv != rocRow) {
    ostringstream out;
     
    out<<"PROBLEM with inversion:  rocCol: "
              <<rocCol<<", inverse: "<<col_inv <<",  rocRow: "<<rocRow<<", inverse: "<<row_inv;
    LogDebug("** HERE**")<<out.str();
  }
*/

  return result;
}


PixelROC::LocalPixel PixelROC::toLocal( const GlobalPixel& glo) const
{
  FrameConversion conversion(theDetUnit, theIdDU);
//    FrameConversion conversion(theRowOffset,theRowSlopeSign,theColOffset,theColSlopeSign);
   int rocRow = conversion.row().inverse(glo.row);
   int rocCol = conversion.collumn().inverse(glo.col);

  LocalPixel loc = {-1,-1};
  if (0<= rocRow && rocRow < PixelROC::rows() && 0 <= rocCol && rocCol <PixelROC::cols()) {
    loc.dcol = rocCol/2;
    loc.pxid = (rocCol %2 == 0) ? rocRow : 2*theNRows - rocRow - 1;
  }
  {
    ostringstream out;
    out<<" detunit: "<<theDetUnit<<" ROC_id: "<<theIdDU;
    out<<" initical global row: "<<glo.row<<" column: "<<glo.col;
    out<<" intermediate local (in ROC frame) rocRow: "<<rocRow<<", rocCol: "<<rocCol;
    out<<" final local pxid: "<<loc.pxid<<" final dcol: "<<loc.dcol;
    LogDebug("** HERE, to Local**")<<out.str();
  }
  return loc;
}



string PixelROC::print(int depth) const
{
  ostringstream out;
  if (depth-- >=0 ) {
    out <<"======== PixelROC " 
        <<" unit: "<<theDetUnit
        <<" idInDU: "<<theIdDU
        <<" idInLk: "<<theIdLk
        <<" RocInXOffset: "<<theRocInX
        <<" RocInYOffset: "<<theRocInY 
        <<endl;     
  }
  return out.str();
}


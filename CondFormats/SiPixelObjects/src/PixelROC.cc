#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include <sstream>
using namespace std;
using namespace sipixelobjects;

int PixelROC::theNRows = 80;
int PixelROC::theNCols = 52;

PixelROC::PixelROC(uint32_t du, int idDU, int idLk, int rocInX, int rocInY)
  : theDetUnit(du), theIdDU(idDU), theIdLk(idLk),
    theRocInX(rocInX), theRocInY(rocInY)
{ }

bool PixelROC::inside(const LocalPixel & lp) const 
{
  return (     0 <= lp.dcol && lp.dcol < theNCols/2
           &&  0 <= lp.pxid && lp.pxid < 2*theNRows );
}

bool PixelROC::inside( int dcol, int pxid) const 
{
  return (     0 <= dcol && dcol < theNCols/2
           &&  0 <= pxid && pxid < 2*theNRows );
}

PixelROC::LocalPixel PixelROC::toLocal( const GlobalPixel& glo) const
{
  LocalPixel loc;
  int rowRoc = glo.row / theNRows; 
  int colRoc = glo.col / theNCols;

  if (rowRoc != theRocInY || colRoc != theRocInX) {
    loc.dcol = -1;
    loc.pxid = -1;
    return loc;
  } 

  int inRow = glo.row % theNRows;
  int inCol = glo.col % theNCols;
  int icol = inCol % 2;

  loc.dcol = inCol / 2;
  loc.pxid = (icol == 0) ? inRow : 2*theNRows - inRow - 1;

  return loc;
}

string PixelROC::print(int depth) const
{
  ostringstream out;
  if (depth-- >=0 ) {
    out <<"======== PixelROC " 
        <<" idInDU: "<<theIdDU
        <<" idInLk: "<<theIdLk
        <<" RocInXOffset: "<<theRocInX
        <<" RocInYOffset: "<<theRocInY 
        <<endl;     
  }
  return out.str();

}


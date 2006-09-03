#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

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
{ }

bool PixelROC::inside( int dcol, int pxid) const 
{
  return (     0 <= dcol && dcol < theNCols/2
           &&  0 <= pxid && pxid < 2*theNRows );
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


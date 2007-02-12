#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/FrameConversion.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
using namespace std;
using namespace sipixelobjects;

int PixelROC::theNRows = 80;
int PixelROC::theNCols = 52;

PixelROC::PixelROC(uint32_t du, int idDU, int idLk, const FrameConversion & frame)
  : theDetUnit(du), 
    theIdDU(idDU), theIdLk(idLk), 
    theRowOffset( frame.row().offset()),     theRowSlopeSign( frame.row().slope()),
    theColOffset( frame.collumn().offset()), theColSlopeSign( frame.collumn().slope())
{ }

bool PixelROC::inside( const LocalPixel & lp) const
{
  return (     0 <= lp.dcol && lp.dcol < 26
           &&  2 <= lp.pxid && lp.pxid < 162 );
}

PixelROC::LocalPixel PixelROC::toLocal( const GlobalPixel& glo) const
{
   FrameConversion conversion(theRowOffset,theRowSlopeSign,theColOffset,theColSlopeSign);
   int rocRow = conversion.row().inverse(glo.row);
   int rocCol = conversion.collumn().inverse(glo.col);

  LocalPixel loc = {-1,-1};
  if (0<= rocRow && rocRow < PixelROC::rows() && 0 <= rocCol && rocCol <PixelROC::cols()) {
    loc.dcol = rocCol/2;
    loc.pxid = 2*(theNRows-rocRow)+ (rocCol%2);
  }
  return loc;
}



string PixelROC::print(int depth) const
{
  ostringstream out;
  bool barrel = ( 1==((theDetUnit>>25)&0x7));
  DetId detId(theDetUnit);
  if (depth-- >=0 ) {
    out <<"======== PixelROC ";
    out <<" unit: ";
    if (barrel) out << PixelBarrelName(detId).name();
    else        out << PixelEndcapName(detId).name(); 
    out <<" ("<<theDetUnit<<")"
        <<" idInDU: "<<theIdDU
        <<" idInLk: "<<theIdLk
        <<" frame: "<<theRowOffset<<","<<theRowSlopeSign<<","<<theColOffset<<","<<theColSlopeSign
        <<endl;
  }
  return out.str();
}


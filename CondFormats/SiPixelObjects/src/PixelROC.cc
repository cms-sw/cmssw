#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
using namespace std;
using namespace sipixelobjects;


PixelROC::PixelROC(uint32_t du, int idDU, int idLk)
  : theDetUnit(du), theIdDU(idDU), theIdLk(idLk), theFrameConverter(0)
{ }

GlobalPixel PixelROC::toGlobal(const LocalPixel & loc) const 
{
  GlobalPixel result;
  if (!theFrameConverter) initFrameConversion();
  result.col    = theFrameConverter->collumn().convert(loc.rocCol());
  result.row    = theFrameConverter->row().convert(loc.rocRow());
  return result;
}


LocalPixel PixelROC::toLocal( const GlobalPixel& glo) const
{
  if (!theFrameConverter) initFrameConversion();
  int rocRow = theFrameConverter->row().inverse(glo.row);
  int rocCol = theFrameConverter->collumn().inverse(glo.col);

  LocalPixel::RocRowCol rocRowCol = {rocRow, rocCol};
  return LocalPixel(rocRowCol);
}

void PixelROC::initFrameConversion() const
{
  if ( PixelModuleName::isBarrel(theDetUnit) ) {
    PixelBarrelName barrelName(theDetUnit); 
    theFrameConverter = new FrameConversion(barrelName, theIdDU);
  } else {
    PixelEndcapName endcapName(theDetUnit);
    theFrameConverter = new FrameConversion(endcapName, theIdDU); 
  }
}

string PixelROC::print(int depth) const
{
  if (!theFrameConverter) initFrameConversion();

  ostringstream out;
  bool barrel = PixelModuleName::isBarrel(theDetUnit);
  DetId detId(theDetUnit);
  if (depth-- >=0 ) {
    out <<"======== PixelROC ";
    out <<" unit: ";
    if (barrel) out << PixelBarrelName(detId).name();
    else        out << PixelEndcapName(detId).name(); 
    out <<" ("<<theDetUnit<<")"
        <<" idInDU: "<<theIdDU
        <<" idInLk: "<<theIdLk
//        <<" frame: "<<theRowOffset<<","<<theRowSlopeSign<<","<<theColOffset<<","<<theColSlopeSign
//        <<" frame: "<<*theFrameConverter
        <<endl;
  }
  return out.str();
}


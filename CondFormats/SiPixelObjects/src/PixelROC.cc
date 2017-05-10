#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
#include <algorithm>
using namespace std;
using namespace sipixelobjects;

PixelROC::PixelROC(uint32_t du, int idDU, int idLk)
  : theDetUnit(du), theIdDU(idDU), theIdLk(idLk)
{initFrameConversion();}

void PixelROC::initFrameConversion()
{
  if ( PixelModuleName::isBarrel(theDetUnit) ) {
    PixelBarrelName barrelName(theDetUnit); 
    theFrameConverter = FrameConversion(barrelName, theIdDU);
  } else {
    PixelEndcapName endcapName(theDetUnit);
    theFrameConverter =  FrameConversion(endcapName, theIdDU); 
  }
}

string PixelROC::print(int depth) const
{

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


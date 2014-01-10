#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
#include <algorithm>
using namespace std;
using namespace sipixelobjects;

PixelROC::PixelROC(uint32_t du, int idDU, int idLk)
  : theDetUnit(du), theIdDU(idDU), theIdLk(idLk), theFrameConverter(0)
{}

PixelROC::PixelROC(const PixelROC & o)
  : theDetUnit(o.theDetUnit), theIdDU(o.theIdDU), theIdLk(o.theIdLk),theFrameConverter(0)
{
  if(o.theFrameConverter) theFrameConverter = o.theFrameConverter->clone();
}

PixelROC::~PixelROC() 
{
  delete theFrameConverter;
}

const PixelROC&
PixelROC::operator=(const PixelROC& iRHS)
{
  PixelROC temp(iRHS);
  this->swap(temp);
  return *this;
}

void
PixelROC::swap(PixelROC& iOther)
{
  std::swap(theDetUnit,iOther.theDetUnit);
  std::swap(theIdDU,iOther.theIdDU);
  std::swap(theIdLk,iOther.theIdLk);
  std::swap(theFrameConverter,iOther.theFrameConverter);
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


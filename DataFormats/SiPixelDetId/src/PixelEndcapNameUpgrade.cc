#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"

#include <sstream>

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;




PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(const DetId & id)
  : PixelModuleName(false)
{
  PXFDetId cmssw_numbering(id);
  int side = cmssw_numbering.side();
  int tmpBlade = cmssw_numbering.blade();
  bool outer = false;
  
/*
  //    iasonas1-with this we determine inner-outer ring, NOT x-direction
  if (tmpBlade >= 23 && tmpBlade <= 56) {       //23-56-->outer ring
    outer = true;
    theBlade = 57-tmpBlade;
  } else if( tmpBlade <= 22 ) {                 //1-22-->inner ring
    theBlade = 23-tmpBlade; 
  } //  iasonas1-end
  
*/
  //    iasonas2-with this we determine inner-outer x-position (x>0-x<0), NOT ring
  //    m/pI:1-6,18-31,49-56. m/pO:7-17,32-48.
  if (tmpBlade>=7 && tmpBlade<=17) {
    outer = true;
    theBlade = tmpBlade-6;                      //7...17-->1...11
  } else if (tmpBlade>=32 && tmpBlade<=48) {
    outer = true;
    theBlade = 60-tmpBlade;                     //32...48-->28...12
  } else if( tmpBlade<=6 ) {
    theBlade = 7-tmpBlade;                      //1...6-->6...1
  } else if( tmpBlade>=18 && tmpBlade<=31 ) {
    theBlade = 38-tmpBlade;                     //18...31-->20...7
  } else if( tmpBlade>=49 && tmpBlade<=56 ) {
    theBlade = 77-tmpBlade;                     //49...56-->28...21
  } //  iasonas2-end
  
       if( side == 1 &&  outer ) thePart = mO;
  else if( side == 1 && !outer ) thePart = mI;
  else if( side == 2 &&  outer ) thePart = pO;
  else if( side == 2 && !outer ) thePart = pI;
  
  theDisk = cmssw_numbering.disk();
  thePannel = cmssw_numbering.panel();
  thePlaquette = cmssw_numbering.module();
  
}


// constructor from name string
PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string name)
  : PixelModuleName(false), thePart(mO), theDisk(0), 
    theBlade(0), thePannel(0), thePlaquette(0) {
  // parse the name string
  // first, check to make sure this is an FPix name, should start with "FPix_"
  // also check to make sure the needed parts are present
  if ( (name.substr(0, 5) != "FPix_") ||
       (name.find("_B") == string::npos) || 
       (name.find("_D") == string::npos) ||
       (name.find("_BLD") == string::npos) || 
       (name.find("_PNL") == string::npos) ||
       (name.find("_PLQ") == string::npos) ) {
    edm::LogError ("BadNameString|SiPixel") 
      << "Bad name string in PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string): "
      << name;
    return;
  }
  // strip off ROC part if it's there
  if (name.find("_ROC") != string::npos)
    name = name.substr(0, name.find("_ROC"));
  // get the half cylinder
  string hcString = name.substr(name.find("_B")+2, name.find("_D")-name.find("_B")-2);
  if (hcString == "mO") thePart = mO;
  else if (hcString == "mI") thePart = mI;
  else if (hcString == "pO") thePart = pO;
  else if (hcString == "pI") thePart = pI;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine half cylinder in PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string): "
      << name;
  }
  // get the disk
  string diskString = name.substr(name.find("_D")+2, name.find("_BLD")-name.find("_D")-2);
  if (diskString == "1") theDisk = 1;
  else if (diskString == "2") theDisk = 2;
  else if (diskString == "3") theDisk = 3;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine disk number in PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string): "
      << name;
  }
  // get the blade
  string bladeString = name.substr(name.find("_BLD")+4, name.find("_PNL")-name.find("_BLD")-4);
  // since atoi() doesn't report errors, do it the long way
  if (bladeString == "1") theBlade = 1;
  else if (bladeString == "2") theBlade = 2;
  else if (bladeString == "3") theBlade = 3;
  else if (bladeString == "4") theBlade = 4;
  else if (bladeString == "5") theBlade = 5;
  else if (bladeString == "6") theBlade = 6;
  else if (bladeString == "7") theBlade = 7;
  else if (bladeString == "8") theBlade = 8;
  else if (bladeString == "9") theBlade = 9;
  else if (bladeString == "10") theBlade = 10;
  else if (bladeString == "11") theBlade = 11;
  else if (bladeString == "12") theBlade = 12;
  else if (bladeString == "13") theBlade = 13;
  else if (bladeString == "14") theBlade = 14;
  else if (bladeString == "15") theBlade = 15;
  else if (bladeString == "16") theBlade = 16;
  else if (bladeString == "17") theBlade = 17;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine blade number in PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string): "
      << name;
  }
  // find the panel
  string panelString = name.substr(name.find("_PNL")+4, name.find("_PLQ")-name.find("_PNL")-4);
  if (panelString == "1") thePannel = 1;
  else if (panelString == "2") thePannel = 2;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine panel number in PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string): "
      << name;
  }
  // find the plaquette
  string plaquetteString = name.substr(name.find("_PLQ")+4, name.size()-name.find("_PLQ")-4);
  if (plaquetteString == "1") thePlaquette = 1;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine plaquette number in PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string): "
      << name;
  }

} // PixelEndcapNameUpgrade::PixelEndcapNameUpgrade(std::string name)


PixelModuleName::ModuleType  PixelEndcapNameUpgrade::moduleType() const
{
  ModuleType type = v2x8;
  if (pannelName() == 1) {
    if (plaquetteName() == 1)      { type = v2x8; }
  }
  else {
    if (plaquetteName() == 1)      { type = v2x8; }
  }
  return type;
}


bool PixelEndcapNameUpgrade::operator== (const PixelModuleName & o) const
{
  if (!o.isBarrel()) {
    const PixelEndcapNameUpgrade * other = dynamic_cast<const PixelEndcapNameUpgrade *>(&o);
    return (    other 
             && thePart      == other->thePart
             && theDisk      == other->theDisk
             && theBlade     == other->theBlade
             && thePannel    == other->thePannel
             && thePlaquette == other->thePlaquette ); 
  } else return false;
}

string PixelEndcapNameUpgrade::name() const 
{
  std::ostringstream stm;
  stm <<"FPix_B"<<thePart<<"_D"<<theDisk<<"_BLD"<<theBlade<<"_PNL"<<thePannel<<"_PLQ"<<thePlaquette;
  return stm.str();
}

std::ostream & operator<<( std::ostream& out, const PixelEndcapNameUpgrade::HalfCylinder& t)
{
  switch (t) {
    case(PixelEndcapNameUpgrade::pI) : {out << "pI"; break;}
    case(PixelEndcapNameUpgrade::pO) : {out << "pO"; break;}
    case(PixelEndcapNameUpgrade::mI) : {out << "mI"; break;}
    case(PixelEndcapNameUpgrade::mO) : {out << "mO"; break;}
    default: out << "unknown";
  };
  return out;
}


// return the DetId
PXFDetId PixelEndcapNameUpgrade::getDetId() {
  
  uint32_t side = 0;
  uint32_t disk = 0;
  uint32_t blade = 0;
  uint32_t panel = 0;
  uint32_t module = 0;

  // figure out the side
  HalfCylinder hc = halfCylinder();
  if (hc == mO || hc == mI) side = 1;
  else if (hc == pO || hc == pI) side = 2;
  
  // get disk/blade/panel/module numbers from PixelEndcapNameUpgrade object
  disk = static_cast<uint32_t>(diskName());
  uint32_t tmpBlade = static_cast<uint32_t>(bladeName());
  panel = static_cast<uint32_t>(pannelName());
  module = static_cast<uint32_t>(plaquetteName());

  // convert blade numbering to cmssw convention
  bool outer = false;
  outer = (hc == mO) || (hc == pO);
/*
  //iasonas1
  if (outer) {blade = 57 - tmpBlade;}
  else       {blade = 23 - tmpBlade;}
  
*/
  //iasonas2    m/pI:1-6,18-31,49-56. m/pO:7-17,32-48.
  if (outer) {
    if          (tmpBlade>=7 && tmpBlade<=17)    theBlade = tmpBlade+6;
    else if     (tmpBlade>=32 && tmpBlade<=48)   theBlade = 60-tmpBlade;
  } else { //inner
    if          (tmpBlade<=6 )                   theBlade = 7-tmpBlade;
    else if     (tmpBlade>=18 && tmpBlade<=31)   theBlade = 38-tmpBlade;
    else if     (tmpBlade>=49 && tmpBlade<=56)   theBlade = 77-tmpBlade;
  }

  // create and return the DetId
  return PXFDetId(side, disk, blade, panel, module);

} // PXFDetId PixelEndcapNameUpgrade::getDetId()










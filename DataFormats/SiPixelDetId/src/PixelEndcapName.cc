#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include <sstream>

#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

PixelEndcapName::PixelEndcapName(const DetId & id)
  : PixelModuleName(false)
{
  PXFDetId cmssw_numbering(id);
  int side = cmssw_numbering.side();

  bool outer = false;
  theBlade = cmssw_numbering.blade();

  if (theBlade >= 1 && theBlade <= 22) {
    outer = true;
  }
       if( side == 1 &&  outer ) thePart = mO;
  else if( side == 1 && !outer ) thePart = mI;
  else if( side == 2 &&  outer ) thePart = pO;
  else if( side == 2 && !outer ) thePart = pI;
 

  theDisk = cmssw_numbering.disk();
  thePannel = cmssw_numbering.panel();
  thePlaquette = cmssw_numbering.module();
}

// constructor from name string
PixelEndcapName::PixelEndcapName(std::string name)
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
      << "Bad name string in PixelEndcapName::PixelEndcapName(std::string): "
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
      << "Unable to determine half cylinder in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }

  // get the disk
  string diskString = name.substr(name.find("_D")+2, name.find("_BLD")-name.find("_D")-2);
  if (diskString == "1") theDisk = 1;
  else if (diskString == "2") theDisk = 2;
  else if (diskString == "3") theDisk = 3;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine disk number in PixelEndcapName::PixelEndcapName(std::string): "
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
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine blade number in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }

  // find the panel
  string panelString = name.substr(name.find("_PNL")+4, name.find("_PLQ")-name.find("_PNL")-4);
  if (panelString == "1") thePannel = 1;
  else if (panelString == "2") thePannel = 2;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine panel number in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }

  // find the plaquette
  string plaquetteString = name.substr(name.find("_PLQ")+4, name.size()-name.find("_PLQ")-4);
  if (plaquetteString == "1") thePlaquette = 1;
  else if (plaquetteString == "2") thePlaquette = 2;
  else if (plaquetteString == "3") thePlaquette = 3;
  else if (plaquetteString == "4") thePlaquette = 4;
  else {
    edm::LogError ("BadNameString|SiPixel") 
      << "Unable to determine plaquette number in PixelEndcapName::PixelEndcapName(std::string): "
      << name;
  }

} // PixelEndcapName::PixelEndcapName(std::string name)

PixelModuleName::ModuleType  PixelEndcapName::moduleType() const
{
  ModuleType type = v1x2;
  if (pannelName() == 1) {
    if (plaquetteName() == 1)      { type = v1x2; }
    else if (plaquetteName() == 2) { type = v2x3; }
    else if (plaquetteName() == 3) { type = v2x4; }
    else if (plaquetteName() == 4) { type = v1x5; }
  }
  else {
    if (plaquetteName() == 1)      { type = v2x3; }
    else if (plaquetteName() == 2) { type = v2x4; }
    else if (plaquetteName() == 3) { type = v2x5; }
  }
  return type;
}

bool PixelEndcapName::operator== (const PixelModuleName & o) const
{
  if (!o.isBarrel()) {
    const PixelEndcapName * other = dynamic_cast<const PixelEndcapName *>(&o);
    return (    other 
             && thePart      == other->thePart
             && theDisk      == other->theDisk
             && theBlade     == other->theBlade
             && thePannel    == other->thePannel
             && thePlaquette == other->thePlaquette ); 
  } else return false;
}


string PixelEndcapName::name() const 
{
  std::ostringstream stm;
  stm <<"FPix_B"<<thePart<<"_D"<<theDisk<<"_BLD"<<theBlade<<"_PNL"<<thePannel<<"_PLQ"<<thePlaquette;
  return stm.str();
}

std::ostream & operator<<( std::ostream& out, const PixelEndcapName::HalfCylinder& t)
{
  switch (t) {
    case(PixelEndcapName::pI) : {out << "pI"; break;}
    case(PixelEndcapName::pO) : {out << "pO"; break;}
    case(PixelEndcapName::mI) : {out << "mI"; break;}
    case(PixelEndcapName::mO) : {out << "mO"; break;}
    default: out << "unknown";
  };
  return out;
}

// return the DetId
PXFDetId PixelEndcapName::getDetId() {
  
  uint32_t side = 0;
  uint32_t disk = 0;
  uint32_t blade = 0;
  uint32_t panel = 0;
  uint32_t module = 0;

  // figure out the side
  HalfCylinder hc = halfCylinder();
  if (hc == mO || hc == mI) side = 1;
  else if (hc == pO || hc == pI) side = 2;
  
  // get disk/blade/panel/module numbers from PixelEndcapName object
  disk = diskName();
  blade = bladeName();
  panel = pannelName();
  module = plaquetteName();

  // create and return the DetId
  return PXFDetId(side, disk, blade, panel, module);

} // PXFDetId PixelEndcapName::getDetId()

#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include <sstream>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

using namespace std;

PixelEndcapName::PixelEndcapName(const DetId & id)
  : PixelModuleName(false)
{
  PXFDetId cmssw_numbering(id);
  int side = cmssw_numbering.side();

  int tmpBlade = cmssw_numbering.blade();
  bool outer = false;
  if (tmpBlade >= 7 && tmpBlade <= 18) {
    outer = true;
    theBlade = tmpBlade-6;
  } else if( tmpBlade <=6 ) { 
    theBlade = 7-tmpBlade; 
  } else if( tmpBlade >= 19) { 
    theBlade = 31-tmpBlade; 
  } 


       if( side == 1 &&  outer ) thePart = mO;
  else if( side == 1 && !outer ) thePart = mI;
  else if( side == 2 &&  outer ) thePart = pO;
  else if( side == 2 && !outer ) thePart = pI;
 

  theDisk = cmssw_numbering.disk();
  thePannel = cmssw_numbering.panel();
  thePlaquette = cmssw_numbering.module();
}

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


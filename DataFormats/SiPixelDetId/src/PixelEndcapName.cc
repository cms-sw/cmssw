#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

#include <sstream>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

using namespace std;

PixelEndcapName::PixelEndcapName(const DetId & id)
  : PixelModuleName(false)
{
  PXFDetId cmssw_numbering(id);
  theEndCap = cmssw_numbering.side();
  theDisk = cmssw_numbering.disk();
  theBlade = cmssw_numbering.blade();
  thePannel = cmssw_numbering.panel();
  thePlaquette = cmssw_numbering.module();
}


string PixelEndcapName::name() const 
{
  std::ostringstream stm;
  stm << "E" << theEndCap;
  stm << "D" << theDisk;
  stm << "B" << theBlade;
  stm << "P" << thePannel;
  stm << "Q" << thePlaquette;
  return stm.str();
}


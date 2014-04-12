#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigSiStrip.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include <iostream>
#include <sstream>

HDQMInspectorConfigSiStrip::HDQMInspectorConfigSiStrip ()
{
}


HDQMInspectorConfigSiStrip::~HDQMInspectorConfigSiStrip ()
{
}


std::string HDQMInspectorConfigSiStrip::translateDetId(const uint32_t id) const
{
  std::stringstream Name;

  uint32_t rawdetid = id;
  SiStripDetId stripdet = SiStripDetId(rawdetid);

  if(stripdet.subDetector() == SiStripDetId::TIB) {
    Name << "TIB";
  } else if(stripdet.subDetector() == SiStripDetId::TID) {
    TIDDetId tid1 = TIDDetId(rawdetid);
    if( tid1.side() == 1 ) {
      Name << "TID-";
    }
    else if( tid1.side() == 2 ) {
      Name << "TID+";
    }
    else {
      Name << "???";
    }
  } else if(stripdet.subDetector() == SiStripDetId::TOB) {
    Name << "TOB";
  } else if( stripdet.subDetector() == SiStripDetId::TEC) {
    TECDetId tec1 = TECDetId(rawdetid);
    if( tec1.side() == 1 ) {
      Name << "TEC-";
    }
    else if( tec1.side() == 2 ) {
      Name << "TEC+";
    }
    else {
      Name << "???";
    }
  } else{
    Name << "???";
  }

  return Name.str();

}

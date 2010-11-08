#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigSiStrip.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"


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
    TIBDetId tib1 = TIBDetId(rawdetid);
    Name << "TIB";
  } else if(stripdet.subDetector() == SiStripDetId::TID) {
    TIDDetId tid1 = TIDDetId(rawdetid);
    Name << "TID";
  } else if(stripdet.subDetector() == SiStripDetId::TOB) {
    TOBDetId tob1 = TOBDetId(rawdetid);
    Name << "TOB";
  } else if( stripdet.subDetector() == SiStripDetId::TEC) {
    TECDetId tec1 = TECDetId(rawdetid);
    Name << "TEC";
  } else{
    Name << "???";
  }

  return Name.str();

}

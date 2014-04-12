#include "DQM/SiStripHistoricInfoClient/interface/HDQMInspectorConfigTracking.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"


#include <sstream>

HDQMInspectorConfigTracking::HDQMInspectorConfigTracking ()
{
}


HDQMInspectorConfigTracking::~HDQMInspectorConfigTracking ()
{
}


std::string HDQMInspectorConfigTracking::translateDetId(const uint32_t id) const
{
  std::stringstream Name;

  if(id == 268435456) {
    Name << "Tracker";
  } else{
    Name << "???";
  }

  return Name.str();

}

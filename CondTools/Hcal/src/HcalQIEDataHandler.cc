#include "CondTools/Hcal/interface/HcalQIEDataHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalQIEDataHandler::HcalQIEDataHandler(edm::ParameterSet const & ps)
{
  m_name = ps.getUntrackedParameter<std::string>("name","HcalQIEDataHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun",0);
}

HcalQIEDataHandler::~HcalQIEDataHandler()
{
}

void HcalQIEDataHandler::getNewObjects()
{
  //  edm::LogInfo   ("HcalQIEDataHandler") 
  std::cout
    << "------- " << m_name 
    << " - > getNewObjects\n" << 
    //check whats already inside of database
    "got offlineInfo"<<
    tagInfo().name << ", size " << tagInfo().size 
					  << ", last object valid since " 
					  << tagInfo().lastInterval.first << std::endl;  

  if (!myDBObject) 
    throw cms::Exception("Empty DB object") << m_name 
					    << " has received empty object - nothing to write to DB" 
					    << std::endl;

  //  IOV information
  cond::Time_t myTime = sinceTime;

  std::cout << "Using IOV run " << sinceTime << std::endl;

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair(myDBObject,myTime));

  edm::LogInfo("HcalQIEDataHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

}

void HcalQIEDataHandler::initObject(HcalQIEData* fObject)
{
  myDBObject = fObject;
}

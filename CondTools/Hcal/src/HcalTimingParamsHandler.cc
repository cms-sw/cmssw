#include "CondTools/Hcal/interface/HcalTimingParamsHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalTimingParamsHandler::HcalTimingParamsHandler(edm::ParameterSet const & ps)
{
  m_name = ps.getUntrackedParameter<std::string>("name","HcalTimingParamsHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun",0);
}

HcalTimingParamsHandler::~HcalTimingParamsHandler()
{
}

void HcalTimingParamsHandler::getNewObjects()
{
  //  edm::LogInfo   ("HcalTimingParamsHandler") 
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

  edm::LogInfo("HcalTimingParamsHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

}

void HcalTimingParamsHandler::initObject(HcalTimingParams* fObject)
{
  myDBObject = fObject;
}

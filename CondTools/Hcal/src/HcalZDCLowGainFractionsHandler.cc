#include "CondTools/Hcal/interface/HcalZDCLowGainFractionsHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalZDCLowGainFractionsHandler::HcalZDCLowGainFractionsHandler(edm::ParameterSet const & ps)
{
  m_name = ps.getUntrackedParameter<std::string>("name","HcalZDCLowGainFractionsHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun",0);
}

HcalZDCLowGainFractionsHandler::~HcalZDCLowGainFractionsHandler()
{
}

void HcalZDCLowGainFractionsHandler::getNewObjects()
{
  edm::LogInfo("HcalZDCLowGainFractionsHandler")
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

  edm::LogInfo("HcalZDCLowGainFractionsHandler") << "Using IOV run " << sinceTime << std::endl;

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair(myDBObject,myTime));

  edm::LogInfo("HcalZDCLowGainFractionsHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

}

void HcalZDCLowGainFractionsHandler::initObject(HcalZDCLowGainFractions* fObject)
{
  myDBObject = fObject;
}

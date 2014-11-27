#include "CondTools/Hcal/interface/HcalODFCorrectionsHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalODFCorrectionsHandler::HcalODFCorrectionsHandler(edm::ParameterSet const & ps)
{
  m_name = ps.getUntrackedParameter<std::string>("name","HcalODFCorrectionsHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun",0);
}

HcalODFCorrectionsHandler::~HcalODFCorrectionsHandler()
{
}

void HcalODFCorrectionsHandler::getNewObjects()
{

  edm::LogInfo("HcalODFCorrectionsHandler")
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

  edm::LogInfo("HcalODFCorrectionsHandler") << "Using IOV run " << sinceTime << std::endl;

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair(myDBObject,myTime));

  edm::LogInfo("HcalODFCorrectionsHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;

}

void HcalODFCorrectionsHandler::initObject(HcalODFCorrections* fObject)
{
  myDBObject = fObject;
}


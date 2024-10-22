#include "CondTools/Hcal/interface/HcalTPParametersHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalTPParametersHandler::HcalTPParametersHandler(edm::ParameterSet const& ps) {
  m_name = ps.getUntrackedParameter<std::string>("name", "HcalTPParametersHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun", 0);
}

HcalTPParametersHandler::~HcalTPParametersHandler() {}

void HcalTPParametersHandler::getNewObjects() {
  edm::LogInfo("HcalCondTools") << "------- " << m_name << " - > getNewObjects\n"
                                << "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size
                                << ", last object valid since " << tagInfo().lastInterval.since << std::endl;

  if (!myDBObject)
    throw cms::Exception("Empty DB object")
        << m_name << " has received empty object - nothing to write to DB" << std::endl;

  //  IOV information
  cond::Time_t myTime = sinceTime;

  edm::LogInfo("HcalCondTools") << "Using IOV run " << sinceTime << std::endl;

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair(myDBObject, myTime));

  edm::LogInfo("HcalCondTools") << "------- " << m_name << " - > getNewObjects" << std::endl;
}

void HcalTPParametersHandler::initObject(HcalTPParameters* fObject) { myDBObject = fObject; }

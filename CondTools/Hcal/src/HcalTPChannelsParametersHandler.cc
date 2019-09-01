#include "CondTools/Hcal/interface/HcalTPChannelParametersHandler.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include <memory>

HcalTPChannelParametersHandler::HcalTPChannelParametersHandler(edm::ParameterSet const& ps) {
  m_name = ps.getUntrackedParameter<std::string>("name", "HcalTPChannelParametersHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun", 0);
}

HcalTPChannelParametersHandler::~HcalTPChannelParametersHandler() {}

void HcalTPChannelParametersHandler::getNewObjects() {
  edm::LogInfo("HcalCondTools") << "------- " << m_name << " - > getNewObjects\n"
                                << "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size
                                << ", last object valid since " << tagInfo().lastInterval.first << std::endl;

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

void HcalTPChannelParametersHandler::initObject(HcalTPChannelParameters* fObject) { myDBObject = fObject; }

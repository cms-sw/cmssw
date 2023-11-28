#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalPFCutsHandler::HcalPFCutsHandler(edm::ParameterSet const& ps) {
  m_name = ps.getUntrackedParameter<std::string>("name", "HcalPFCutsHandler");
  sinceTime = ps.getUntrackedParameter<unsigned>("IOVRun", 0);
}

HcalPFCutsHandler::~HcalPFCutsHandler() {}

void HcalPFCutsHandler::getNewObjects() {
  edm::LogInfo("HcalPFCutsHandler") << "------- " << m_name << " - > getNewObjects\n"
                                    <<
      //check whats already inside of database
      "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size << ", last object valid since "
                                    << tagInfo().lastInterval.since;

  if (!myDBObject)
    throw cms::Exception("Empty DB object") << m_name << " has received empty object - nothing to write to DB";

  //  IOV information
  cond::Time_t myTime = sinceTime;

  edm::LogInfo("HcalPFCutsHandler") << "Using IOV run " << sinceTime;

  // prepare for transfer:
  m_to_transfer.push_back(std::make_pair(myDBObject, myTime));

  edm::LogInfo("HcalPFCutsHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;
}

void HcalPFCutsHandler::initObject(HcalPFCuts* fObject) { myDBObject = fObject; }

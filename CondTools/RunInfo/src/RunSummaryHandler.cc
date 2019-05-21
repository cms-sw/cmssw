#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/RunSummaryHandler.h"
#include "CondTools/RunInfo/interface/RunSummaryRead.h"

#include <iostream>
#include <vector>

RunSummaryHandler::RunSummaryHandler(const edm::ParameterSet& pset)
    : m_name(pset.getUntrackedParameter<std::string>("name", "RunSummaryHandler")),
      // m_connect(pset.getUntrackedParameter<std::string>("OnlineConn","")),

      m_user(pset.getUntrackedParameter<std::string>("OnlineDBUser", "CMS_RUNINFO")),
      m_pass(pset.getUntrackedParameter<std::string>("OnlineDBPass", "********"))

{
  m_connectionString = "oracle://cms_omds_lb/CMS_RUNINFO";
}

RunSummaryHandler::~RunSummaryHandler() {}

void RunSummaryHandler::getNewObjects() {
  edm::LogInfo("RunSummaryHandler") << "------- " << m_name << " - > getNewObjects\n"
                                    <<
      //check whats already inside of database
      "got offlineInfo" << tagInfo().name << ", size " << tagInfo().size << ", last object valid since "
                                    << tagInfo().lastInterval.first << " token " << tagInfo().lastPayloadToken
                                    << std::endl;

  /*
     if (tagInfo().size>0) {
     Ref payload = lastPayload();
     //edm::LogInfo   ("RunSummaryHandler")<<"size of last payload  "<< 
     //   payload->..........size()<<std::endl;
     }
   */
  unsigned int snc;

  std::cerr << "Source implementation test ::getNewObjects : enter runnumber as a first since !  \n";
  std::cin >> snc;

  std::cout << "runnumber/first since = " << snc << std::endl;

  RunSummary* r = new RunSummary();

  //fill with null runsummary if empty run are found beetween the two last validones

  size_t n_empty_run = 0;
  if (tagInfo().size > 0 && (tagInfo().lastInterval.first + 1) < snc) {
    n_empty_run = snc - tagInfo().lastInterval.first - 1;
    edm::LogInfo("RunSummaryHandler") << "------- "
                                      << "entering fake run from " << tagInfo().lastInterval.first + 1 << "to "
                                      << snc - 1 << "- > getNewObjects" << std::endl;
    n_empty_run = snc - tagInfo().lastInterval.first - 1;
    // for (size_t i=1; i<= n_empty_run ; i++){

    // r->summary.push_back(empty->fake_Run());
    // }
  }

  // transfer fake run for 1 to since for the first time
  if (tagInfo().size == 0) {
    m_to_transfer.push_back(std::make_pair((RunSummary*)(r->Fake_RunSummary()), 1));
  }

  // transfer also empty run if tag already existing
  if (n_empty_run != 0) {
    m_to_transfer.push_back(std::make_pair((RunSummary*)(r->Fake_RunSummary()), tagInfo().lastInterval.first + 1));
  }

  // reading from omds
  RunSummaryRead rn(m_connectionString, m_user, m_pass);

  *r = rn.readData("RUNSESSION_PARAMETER", "STRING_VALUE", (int)snc);
  m_to_transfer.push_back(std::make_pair((RunSummary*)r, snc));
  std::ostringstream ss;
  ss << "since =" << snc;

  m_userTextLog = ss.str() + ";";

  edm::LogInfo("RunSummaryHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;
}

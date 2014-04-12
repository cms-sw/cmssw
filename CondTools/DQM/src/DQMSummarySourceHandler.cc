#include "CondTools/DQM/interface/DQMSummarySourceHandler.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondTools/DQM/interface/DQMSummaryReader.h"
#include <iostream>
#include <string>
#include <vector>

namespace popcon {
  DQMSummarySourceHandler::DQMSummarySourceHandler(const edm::ParameterSet & pset):
    m_name(pset.getUntrackedParameter<std::string>("name","DQMSummarySourceHandler")),
    m_since(pset.getUntrackedParameter<unsigned long long>("firstSince",1)), 
    m_user(pset.getUntrackedParameter<std::string>("OnlineDBUser","CMS_DQM_SUMMARY")), 
    m_pass(pset.getUntrackedParameter<std::string>("OnlineDBPass","****")) {
    m_connectionString = "oracle://cms_omds_lb/CMS_DQM_SUMMARY";
  }
  
  DQMSummarySourceHandler::~DQMSummarySourceHandler() {}
  
  void DQMSummarySourceHandler::getNewObjects() {
    //check what is already inside of the database
    edm::LogInfo("DQMSummarySourceHandler") << "------- " << m_name << " -> getNewObjects\n" 
					    << "got offlineInfo " << tagInfo().name 
					    << ", size " << tagInfo().size 
					    << ", last object valid since " 
					    << tagInfo().lastInterval.first << " token "   
					    << tagInfo().lastPayloadToken << std::endl;
    edm::LogInfo("DQMSummarySourceHandler") << " ------ last entry info regarding the payload (if existing): " 
					    << logDBEntry().usertext
					    << "; last record with the correct tag (if existing) has been written in the db: " 
					    << logDBEntry().destinationDB << std::endl; 
    if (tagInfo().size > 0) {
      Ref payload = lastPayload();
      edm::LogInfo("DQMSummarySourceHandler") << "size of last payload  "
					      << payload->m_summary.size() << std::endl;
    }
    std::cout << "runnumber/first since = " << m_since << std::endl;
    DQMSummary * dqmSummary = new DQMSummary;
    DQMSummaryReader dqmSummaryReader(m_connectionString, m_user, m_pass);
    *dqmSummary = dqmSummaryReader.readData("SUMMARYCONTENT", m_since);
    m_to_transfer.push_back(std::make_pair((DQMSummary*)dqmSummary,m_since));
    edm::LogInfo("DQMSummarySourceHandler") << "------- " 
					    << m_name << " - > getNewObjects" 
					    << std::endl;
  }
  
  std::string DQMSummarySourceHandler::id() const {return m_name;}
}

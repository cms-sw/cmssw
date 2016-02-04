#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RunInfo/interface/RunInfoHandler.h"
#include "CondTools/RunInfo/interface/RunInfoRead.h"
#include<iostream>
#include<vector>

RunInfoHandler::RunInfoHandler(const edm::ParameterSet& pset) :
  m_name(pset.getUntrackedParameter<std::string>("name","RunInfoHandler"))
  ,m_user(pset.getUntrackedParameter<std::string>("OnlineDBUser","CMS_RUNINFO_R")) 
  ,m_pass(pset.getUntrackedParameter<std::string>("OnlineDBPass","PASSWORD")) {
  m_connectionString= "oracle://cms_omds_lb/CMS_RUNINFO";
}

RunInfoHandler::~RunInfoHandler() {}

void RunInfoHandler::getNewObjects() {
  //check whats already inside of database
  edm::LogInfo   ("RunInfoHandler") << "------- " << m_name 
				    << " - > getNewObjects\n" 
				    << "got offlineInfo " << tagInfo().name 
				    << ", size " << tagInfo().size 
				    << ", last object valid since " 
				    << tagInfo().lastInterval.first 
				    << " token " << tagInfo().lastPayloadToken 
				    << std::endl;
  unsigned int snc;
  std::cerr << "Source implementation test ::getNewObjects : enter runnumber as a first since !\n";
  std::cin >> snc;
  std::cout <<"runnumber/first since = " << snc << std::endl;
  RunInfo* r = new RunInfo(); 
  
  //fill with null runinfo if empty run are found beetween the two last valid ones 
  size_t n_empty_run = 0;
  if(tagInfo().size > 0  && (tagInfo().lastInterval.first+1) < snc) {
    n_empty_run = snc - tagInfo().lastInterval.first - 1; 
    edm::LogInfo   ("RunInfoHandler") << "------- " << "entering fake run from " 
				      << tagInfo().lastInterval.first + 1 
				      <<  "to " << snc - 1 << "- > getNewObjects" 
				      << std::endl;
    n_empty_run = snc - tagInfo().lastInterval.first - 1; 
  } 
  // transfer fake run for 1 to since for the first time
  if (tagInfo().size == 0 && snc != 1) {
    m_to_transfer.push_back(std::make_pair((RunInfo*) (r->Fake_RunInfo()),1));
  }
  if (n_empty_run != 0) {
    m_to_transfer.push_back(std::make_pair((RunInfo*) (r->Fake_RunInfo()),tagInfo().lastInterval.first + 1));
  }
  
  //reading from omds
  RunInfoRead rn(m_connectionString, m_user, m_pass);
  *r = rn.readData("RUNSESSION_PARAMETER", "STRING_VALUE",(int)snc);
  m_to_transfer.push_back(std::make_pair((RunInfo*)r,snc));
  std::ostringstream ss;
  ss << "since =" << snc;
  m_userTextLog = ss.str() + ";";
  edm::LogInfo   ("RunInfoHandler") << "------- " << m_name << " - > getNewObjects" << std::endl;
}

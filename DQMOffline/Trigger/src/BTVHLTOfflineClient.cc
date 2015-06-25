// 

#include "DQMOffline/Trigger/interface/BTVHLTOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

BTVHLTOfflineClient::BTVHLTOfflineClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  debug_ = false;
  verbose_ = false;

  processname_ = iConfig.getParameter<std::string>("processname");

  hltTag_ = iConfig.getParameter<std::string>("hltTag");
  if (debug_) std::cout << hltTag_ << std::endl;
  
  dirName_=iConfig.getParameter<std::string>("DQMDirName");
 
}

BTVHLTOfflineClient::~BTVHLTOfflineClient()
{ 
  
}

void BTVHLTOfflineClient::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter)
{
}


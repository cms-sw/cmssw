#include "DQMOffline/JetMET/interface/JetMETDQMOfflineClient.h"

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/MonitorElement.h"

JetMETDQMOfflineClient::JetMETDQMOfflineClient(const edm::ParameterSet& iConfig):conf_(iConfig)
{
  verbose_   = conf_.getUntrackedParameter<int>("Verbose");
  dirName_   =iConfig.getUntrackedParameter<std::string>("DQMDirName");
  dirNameJet_=iConfig.getUntrackedParameter<std::string>("DQMJetDirName");
}


JetMETDQMOfflineClient::~JetMETDQMOfflineClient()
{ 
}

void JetMETDQMOfflineClient::dqmEndJob(DQMStore::IBooker & ibook_, DQMStore::IGetter & iget_) 
{
  //empty client. is this wanted?
}


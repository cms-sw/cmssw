#include "../interface/EcalDQMonitorClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

#include <ctime>
#include <fstream>

EcalDQMonitorClient::EcalDQMonitorClient(edm::ParameterSet const& _ps) :
  edm::EDAnalyzer(),
  ecaldqm::EcalDQMonitor(_ps),
  eventCycleLength_(_ps.getUntrackedParameter<unsigned>("analyzeEvery")),
  iEvt_(0),
  statusManager_()
{
  executeOnWorkers_([this](ecaldqm::DQWorker* worker){
      ecaldqm::DQWorkerClient* client(dynamic_cast<ecaldqm::DQWorkerClient*>(worker));
      if(!client)
        throw cms::Exception("InvalidConfiguration") << "Non-client DQWorker " << worker->getName() << " passed";
      client->setStatusManager(this->statusManager_);
    }, "initialization");

  if(_ps.existsAs<edm::FileInPath>("PNMaskFile", false)){
    std::ifstream maskFile(_ps.getUntrackedParameter<edm::FileInPath>("PNMaskFile").fullPath());
    if(maskFile.is_open())
      statusManager_.readFromStream(maskFile);
  }
}

/*static*/
void
EcalDQMonitorClient::fillDescriptions(edm::ConfigurationDescriptions &_descs)
{
  edm::ParameterSetDescription desc;
  ecaldqm::EcalDQMonitor::fillDescriptions(desc);

  edm::ParameterSetDescription clientParameters;
  ecaldqm::DQWorkerClient::fillDescriptions(clientParameters);
  edm::ParameterSetDescription allWorkers;
  allWorkers.addNode(edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, false, clientParameters));
  desc.addUntracked("workerParameters", allWorkers);

  desc.addUntracked<unsigned>("analyzeEvery", 0);
  desc.addOptionalUntracked<edm::FileInPath>("PNMaskFile");

  _descs.addDefault(desc);
}

void
EcalDQMonitorClient::beginRun(edm::Run const& _run, edm::EventSetup const& _es)
{
  ecaldqmGetSetupObjects(_es);

  if(_es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMChannelStatusRcd>()) && _es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMTowerStatusRcd>())){
    edm::ESHandle<EcalDQMChannelStatus> cStHndl;
    _es.get<EcalDQMChannelStatusRcd>().get(cStHndl);

    edm::ESHandle<EcalDQMTowerStatus> tStHndl;
    _es.get<EcalDQMTowerStatusRcd>().get(tStHndl);

    statusManager_.readFromObj(*cStHndl, *tStHndl);
  }

  ecaldqmBookHistograms(*edm::Service<DQMStore>());

  ecaldqmBeginRun(_run, _es);
}

void
EcalDQMonitorClient::endRun(edm::Run const& _run, edm::EventSetup const& _es)
{
  ecaldqmEndRun(_run, _es);

  runWorkers(ecaldqm::DQWorkerClient::kRun);

  ecaldqmReleaseHistograms();
}

void
EcalDQMonitorClient::endLuminosityBlock(edm::LuminosityBlock const& _lumi, edm::EventSetup const& _es)
{
  ecaldqmEndLuminosityBlock(_lumi, _es);

  runWorkers(ecaldqm::DQWorkerClient::kLumi);
}

void
EcalDQMonitorClient::analyze(edm::Event const&, edm::EventSetup const&)
{
  if(eventCycleLength_ != 0 && ++iEvt_ % eventCycleLength_ == 0) runWorkers(ecaldqm::DQWorkerClient::kLumi);
}

void
EcalDQMonitorClient::runWorkers(ecaldqm::DQWorkerClient::ProcessType _type)
{
  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << moduleName_ << ": Starting worker modules..";

  DQMStore const& dqmStore(*edm::Service<DQMStore>());

  executeOnWorkers_([&dqmStore, &_type](ecaldqm::DQWorker* worker){
      ecaldqm::DQWorkerClient* client(static_cast<ecaldqm::DQWorkerClient*>(worker));
      client->releaseSource();
      client->resetMEs();
      if(!client->retrieveSource(dqmStore, _type)) return;
      if(client->onlineMode()) client->setTime(time(0));
      client->producePlots(_type);                      
    }, "retrieveAndRun", "producing plots");

  if(verbosity_ > 0) edm::LogInfo("EcalDQM") << " done." << std::endl;
}

DEFINE_FWK_MODULE(EcalDQMonitorClient);

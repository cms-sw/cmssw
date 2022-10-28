#include "DQM/EcalMonitorClient/interface/EcalDQMonitorClient.h"

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

#include <ctime>
#include <fstream>

EcalDQMonitorClient::EcalDQMonitorClient(edm::ParameterSet const& _ps)
    : DQMEDHarvester(),
      ecaldqm::EcalDQMonitor(_ps),
      iEvt_(0),
      cStHndl(esConsumes<edm::Transition::BeginRun>()),
      tStHndl(esConsumes<edm::Transition::BeginRun>()),
      statusManager_() {
  edm::ConsumesCollector collector(consumesCollector());
  executeOnWorkers_(
      [this, &collector](ecaldqm::DQWorker* worker) {
        ecaldqm::DQWorkerClient* client(dynamic_cast<ecaldqm::DQWorkerClient*>(worker));
        if (!client)
          throw cms::Exception("InvalidConfiguration") << "Non-client DQWorker " << worker->getName() << " passed";
        client->setStatusManager(this->statusManager_);
        client->setTokens(collector);
        worker->setTokens(collector);
      },
      "initialization");

  // This is no longer used since run 2
  //
  //if (_ps.existsAs<edm::FileInPath>("PNMaskFile", false)) {
  //  std::ifstream maskFile(_ps.getUntrackedParameter<edm::FileInPath>("PNMaskFile").fullPath());
  //  if (maskFile.is_open())
  //    statusManager_.readFromStream(maskFile);
  //}
}

EcalDQMonitorClient::~EcalDQMonitorClient() {}

/*static*/
void EcalDQMonitorClient::fillDescriptions(edm::ConfigurationDescriptions& _descs) {
  edm::ParameterSetDescription desc;
  ecaldqm::EcalDQMonitor::fillDescriptions(desc);

  edm::ParameterSetDescription clientParameters;
  ecaldqm::DQWorkerClient::fillDescriptions(clientParameters);
  edm::ParameterSetDescription allWorkers;
  allWorkers.addNode(
      edm::ParameterWildcard<edm::ParameterSetDescription>("*", edm::RequireZeroOrMore, false, clientParameters));
  desc.addUntracked("workerParameters", allWorkers);

  desc.addOptionalUntracked<edm::FileInPath>("PNMaskFile");

  _descs.addDefault(desc);
}

void EcalDQMonitorClient::beginRun(edm::Run const& _run, edm::EventSetup const& _es) {
  executeOnWorkers_([&_es](ecaldqm::DQWorker* worker) { worker->setSetupObjects(_es); },
                    "ecaldqmGetSetupObjects",
                    "Getting EventSetup Objects");

  if (_es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMChannelStatusRcd>()) &&
      _es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMTowerStatusRcd>())) {
    const EcalDQMChannelStatus* ChStatus = &_es.getData(cStHndl);
    const EcalDQMTowerStatus* TStatus = &_es.getData(tStHndl);

    statusManager_.readFromObj(*ChStatus, *TStatus);
  }

  ecaldqmBeginRun(_run, _es);
}

void EcalDQMonitorClient::endRun(edm::Run const& _run, edm::EventSetup const& _es) { ecaldqmEndRun(_run, _es); }

void EcalDQMonitorClient::dqmEndLuminosityBlock(DQMStore::IBooker& _ibooker,
                                                DQMStore::IGetter& _igetter,
                                                edm::LuminosityBlock const& _lumi,
                                                edm::EventSetup const& _es) {
  executeOnWorkers_(
      [&_ibooker](ecaldqm::DQWorker* worker) {
        ecaldqm::DQWorkerClient* client(static_cast<ecaldqm::DQWorkerClient*>(worker));
        if (!client->onlineMode() && !client->runsOn(ecaldqm::DQWorkerClient::kLumi))
          return;
        client->bookMEs(_ibooker);
      },
      "bookMEs",
      "Booking MEs");

  ecaldqmEndLuminosityBlock(_lumi, _es);

  runWorkers(_igetter, ecaldqm::DQWorkerClient::kLumi);

  executeOnWorkers_(
      [](ecaldqm::DQWorker* worker) {
        ecaldqm::DQWorkerClient* client(static_cast<ecaldqm::DQWorkerClient*>(worker));
        client->resetPerLumi();
      },
      "dqmEndLuminosityBlock",
      "Reset per-lumi MEs");
}

void EcalDQMonitorClient::dqmEndJob(DQMStore::IBooker& _ibooker, DQMStore::IGetter& _igetter) {
  executeOnWorkers_(
      [&_ibooker](ecaldqm::DQWorker* worker) {
        if (!worker->checkElectronicsMap(false))  // to avoid crashes on empty runs
          return;
        worker->bookMEs(_ibooker);  // worker returns if already booked
      },
      "bookMEs",
      "Booking MEs");

  runWorkers(_igetter, ecaldqm::DQWorkerClient::kJob);

  executeOnWorkers_([](ecaldqm::DQWorker* worker) { worker->releaseMEs(); }, "releaseMEs", "releasing histograms");
}

void EcalDQMonitorClient::runWorkers(DQMStore::IGetter& _igetter, ecaldqm::DQWorkerClient::ProcessType _type) {
  if (verbosity_ > 0)
    edm::LogInfo("EcalDQM") << moduleName_ << ": Starting worker modules..";

  executeOnWorkers_(
      [&_igetter, &_type](ecaldqm::DQWorker* worker) {
        if (!worker->checkElectronicsMap(false))  // to avoid crashes on empty runs
          return;
        ecaldqm::DQWorkerClient* client(static_cast<ecaldqm::DQWorkerClient*>(worker));
        if (!client->onlineMode() && !client->runsOn(_type))
          return;
        client->releaseSource();
        client->resetMEs();
        if (!client->retrieveSource(_igetter, _type))
          return;
        if (client->onlineMode())
          client->setTime(time(nullptr));
        client->producePlots(_type);
      },
      "retrieveAndRun",
      "producing plots");

  if (verbosity_ > 0)
    edm::LogInfo("EcalDQM") << " done." << std::endl;
}

DEFINE_FWK_MODULE(EcalDQMonitorClient);

#include "DQM/EcalCommon/interface/EcalDQMonitorClient.h"

#include "DQM/EcalCommon/interface/DQWorkerClient.h"
#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

using namespace ecaldqm;

EcalDQMonitorClient::EcalDQMonitorClient(const edm::ParameterSet &_ps) :
  EcalDQMonitor(_ps),
  workers_(0),
  runAtEndLumi_(_ps.getUntrackedParameter<bool>("runAtEndLumi", false)),
  lumiStatus_(-1)
{
  using namespace std;

  const edm::ParameterSet& clientParams(_ps.getUntrackedParameterSet("clientParameters"));
  const edm::ParameterSet& mePaths(_ps.getUntrackedParameterSet("mePaths"));

  vector<string> clientNames(_ps.getUntrackedParameter<vector<string> >("clients"));

  WorkerFactory factory(0);

  for(vector<string>::iterator cItr(clientNames.begin()); cItr != clientNames.end(); ++cItr){
    if (!(factory = SetWorker::findFactory(*cItr))) continue;

    if(verbosity_ > 0) cout << moduleName_ << ": Setting up " << *cItr << endl;

    DQWorker* worker(factory(clientParams, mePaths.getUntrackedParameterSet(*cItr)));
    if(worker->getName() != *cItr){
      delete worker;
      continue;
    }
    DQWorkerClient* client(static_cast<DQWorkerClient*>(worker));
    client->setVerbosity(verbosity_);

    workers_.push_back(client);
  }
}

EcalDQMonitorClient::~EcalDQMonitorClient()
{
  for(std::vector<DQWorkerClient*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    delete *wItr;
}

/* static */
void
EcalDQMonitorClient::fillDescriptions(edm::ConfigurationDescriptions &_descs)
{
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  _descs.addDefault(desc);
}

void
EcalDQMonitorClient::beginRun(const edm::Run &_run, const edm::EventSetup &_es)
{
  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  ecaldqm::setElectronicsMap(elecMapHandle.product());

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
  _es.get<IdealGeometryRecord>().get(ttMapHandle);
  ecaldqm::setTrigTowerMap(ttMapHandle.product());

  if(_es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMChannelStatusRcd>())){
    edm::ESHandle<EcalDQMChannelStatus> cStHndl;
    _es.get<EcalDQMChannelStatusRcd>().get(cStHndl);
    DQWorkerClient::channelStatus = cStHndl.product();
  }

  if(_es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMTowerStatusRcd>())){
    edm::ESHandle<EcalDQMTowerStatus> tStHndl;
    _es.get<EcalDQMTowerStatusRcd>().get(tStHndl);
    DQWorkerClient::towerStatus = tStHndl.product();
  }

  for(std::vector<DQWorkerClient*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerClient* client(*wItr);
    if(verbosity_ > 1) std::cout << moduleName_ << ": Booking MEs for " << client->getName() << std::endl;
    client->bookMEs();
    client->beginRun(_run, _es);
  }

  lumiStatus_ = -1;

  if(verbosity_ > 0)
    std::cout << moduleName_ << ": Starting run " << _run.run() << std::endl;
}

void
EcalDQMonitorClient::endRun(const edm::Run &_run, const edm::EventSetup &_es)
{
  if(lumiStatus_ == 0)
    runWorkers();

  for(std::vector<DQWorkerClient *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    (*wItr)->endRun(_run, _es);
  }
}

void
EcalDQMonitorClient::beginLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  for(std::vector<DQWorkerClient *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    (*wItr)->beginLuminosityBlock(_lumi, _es);

  lumiStatus_ = 0;
}

void
EcalDQMonitorClient::endLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  for(std::vector<DQWorkerClient *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    (*wItr)->endLuminosityBlock(_lumi, _es);

  if(runAtEndLumi_){
    runWorkers();
    lumiStatus_ = 1;
  }
}

void
EcalDQMonitorClient::runWorkers()
{
  if(verbosity_ > 0)
    std::cout << "EcalDQMonitorClient: Starting worker modules.." << std::endl;

  for(std::vector<DQWorkerClient *>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerClient* client(*wItr);
    if(!client->isInitialized())
      client->initialize();

    if(client->isInitialized())
      client->producePlots();
  }

  if(verbosity_ > 0)
    std::cout << " done." << std::endl;
}

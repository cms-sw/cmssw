#include "../interface/EcalDQMonitorClient.h"

#include "../interface/DQWorkerClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include <ctime>

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDQMChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalDQMTowerStatusRcd.h"

using namespace ecaldqm;

EcalDQMonitorClient::EcalDQMonitorClient(const edm::ParameterSet &_ps) :
  EcalDQMonitor(_ps)
{
  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    if(!dynamic_cast<DQWorkerClient*>(*wItr))
      throw cms::Exception("InvalidConfiguration") << "Non-client DQWorker " << (*wItr)->getName() << " passed";

  if(_ps.existsAs<edm::FileInPath>("PNMaskFile", false))
    ecaldqm::readPNMaskMap(_ps.getUntrackedParameter<edm::FileInPath>("PNMaskFile"));
}

EcalDQMonitorClient::~EcalDQMonitorClient()
{
  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
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
  DQWorker::iRun = _run.run();
  DQWorker::now = time(0);

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  ecaldqm::setElectronicsMap(elecMapHandle.product());

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
  _es.get<IdealGeometryRecord>().get(ttMapHandle);
  ecaldqm::setTrigTowerMap(ttMapHandle.product());

  if(_es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMChannelStatusRcd>()) && _es.find(edm::eventsetup::EventSetupRecordKey::makeKey<EcalDQMTowerStatusRcd>())){
    edm::ESHandle<EcalDQMChannelStatus> cStHndl;
    _es.get<EcalDQMChannelStatusRcd>().get(cStHndl);

    edm::ESHandle<EcalDQMTowerStatus> tStHndl;
    _es.get<EcalDQMTowerStatusRcd>().get(tStHndl);

    ecaldqm::setStatuses(cStHndl.product(), tStHndl.product());
  }

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorkerClient* worker(static_cast<DQWorkerClient*>(*wItr));
    if(verbosity_ > 1) std::cout << moduleName_ << ": Booking summary MEs for " << worker->getName() << std::endl;
    worker->bookSummaries();
    worker->beginRun(_run, _es);
  }

  if(verbosity_ > 0)
    std::cout << moduleName_ << ": Starting run " << _run.run() << std::endl;
}

void
EcalDQMonitorClient::endRun(const edm::Run &_run, const edm::EventSetup &_es)
{
  runWorkers();

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    (*wItr)->endRun(_run, _es);
}

void
EcalDQMonitorClient::beginLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  DQWorker::iLumi = _lumi.luminosityBlock();
  DQWorker::now = time(0);

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    (*wItr)->beginLuminosityBlock(_lumi, _es);
}

void
EcalDQMonitorClient::endLuminosityBlock(const edm::LuminosityBlock &_lumi, const edm::EventSetup &_es)
{
  runWorkers();

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr)
    (*wItr)->endLuminosityBlock(_lumi, _es);
}

void
EcalDQMonitorClient::runWorkers()
{
  if(verbosity_ > 0)
    std::cout << moduleName_ << ": Starting worker modules.." << std::endl;

  for(std::vector<DQWorker*>::iterator wItr(workers_.begin()); wItr != workers_.end(); ++wItr){
    DQWorker* worker(*wItr);
    if(!worker->isInitialized()){
      if(verbosity_ > 1)
        std::cout << " initializing " << worker->getName() << std::endl;
      worker->initialize();
    }

    if(worker->isInitialized()){
      if(verbosity_ > 1)
        std::cout << " producing plots in " << worker->getName() << std::endl;
      static_cast<DQWorkerClient*>(worker)->producePlots();
    }
  }

  if(verbosity_ > 0)
    std::cout << " done." << std::endl;
}

DEFINE_FWK_MODULE(EcalDQMonitorClient);

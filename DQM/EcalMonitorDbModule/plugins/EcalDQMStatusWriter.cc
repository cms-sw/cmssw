#include "DQM/EcalMonitorDbModule/interface/EcalDQMStatusWriter.h"

#include "DQM/EcalCommon/interface/StatusManager.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/interface/MakerMacros.h"

EcalDQMStatusWriter::EcalDQMStatusWriter(edm::ParameterSet const &_ps)
    : channelStatus_(),
      towerStatus_(),
      firstRun_(_ps.getUntrackedParameter<unsigned>("firstRun")),
      inputFile_(_ps.getUntrackedParameter<std::string>("inputFile")),
      electronicsMap_(nullptr),
      elecMapHandle_(esConsumes<edm::Transition::BeginRun>()) {
  if (!inputFile_.is_open())
    throw cms::Exception("Invalid input for EcalDQMStatusWriter");
}

void EcalDQMStatusWriter::beginRun(edm::Run const &_run, edm::EventSetup const &_es) {
  setElectronicsMap(_es);

  ecaldqm::StatusManager statusManager;

  statusManager.readFromStream(inputFile_, GetElectronicsMap());
  statusManager.writeToObj(channelStatus_, towerStatus_);
}

void EcalDQMStatusWriter::endRun(edm::Run const &_run, edm::EventSetup const &_es) {}

void EcalDQMStatusWriter::analyze(edm::Event const &, edm::EventSetup const &_es) {
  cond::service::PoolDBOutputService &dbOutput(*edm::Service<cond::service::PoolDBOutputService>());
  if (firstRun_ == dbOutput.endOfTime())
    return;

  dbOutput.writeOneIOV(channelStatus_, firstRun_, "EcalDQMChannelStatusRcd");
  dbOutput.writeOneIOV(towerStatus_, firstRun_, "EcalDQMTowerStatusRcd");

  firstRun_ = dbOutput.endOfTime();  // avoid accidentally re-writing the conditions
}

void EcalDQMStatusWriter::setElectronicsMap(edm::EventSetup const &_es) {
  electronicsMap_ = &_es.getData(elecMapHandle_);
}

EcalElectronicsMapping const *EcalDQMStatusWriter::GetElectronicsMap() {
  if (!electronicsMap_)
    throw cms::Exception("InvalidCall") << "Electronics Mapping not initialized";
  return electronicsMap_;
}

DEFINE_FWK_MODULE(EcalDQMStatusWriter);

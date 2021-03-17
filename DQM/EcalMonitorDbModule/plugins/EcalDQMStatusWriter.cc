#include "DQM/EcalMonitorDbModule/interface/EcalDQMStatusWriter.h"

#include "DQM/EcalCommon/interface/StatusManager.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/Utils/interface/Exception.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>

EcalDQMStatusWriter::EcalDQMStatusWriter(edm::ParameterSet const &_ps)
    : channelStatus_(),
      towerStatus_(),
      firstRun_(_ps.getUntrackedParameter<unsigned>("firstRun")),
      inputFile_(_ps.getUntrackedParameter<std::string>("inputFile")) {
  if (!inputFile_.is_open())
    throw cms::Exception("Invalid input for EcalDQMStatusWriter");
}

void EcalDQMStatusWriter::beginRun(edm::Run const &_run, edm::EventSetup const &_es) {
  setElectronicsMap(_es);

  ecaldqm::StatusManager statusManager;

  statusManager.readFromStream(inputFile_, GetElectronicsMap());
  statusManager.writeToObj(channelStatus_, towerStatus_);
}

void EcalDQMStatusWriter::analyze(edm::Event const &, edm::EventSetup const &_es) {
  cond::service::PoolDBOutputService &dbOutput(*edm::Service<cond::service::PoolDBOutputService>());
  if (firstRun_ == dbOutput.endOfTime())
    return;

  dbOutput.writeOne(&channelStatus_, firstRun_, "EcalDQMChannelStatusRcd");
  dbOutput.writeOne(&towerStatus_, firstRun_, "EcalDQMTowerStatusRcd");

  firstRun_ = dbOutput.endOfTime();  // avoid accidentally re-writing the conditions
}

void EcalDQMStatusWriter::setElectronicsMap(edm::EventSetup const &_es) {
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  electronicsMap = elecMapHandle.product();
}

EcalElectronicsMapping const *EcalDQMStatusWriter::GetElectronicsMap() {
  if (!electronicsMap)
    throw cms::Exception("InvalidCall") << "Electronics Mapping not initialized";
  return electronicsMap;
}

DEFINE_FWK_MODULE(EcalDQMStatusWriter);

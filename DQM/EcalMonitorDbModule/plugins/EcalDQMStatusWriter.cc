#include "../interface/EcalDQMStatusWriter.h"

#include "DQM/EcalCommon/interface/StatusManager.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CommonTools/Utils/interface/Exception.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include <fstream>

EcalDQMStatusWriter::EcalDQMStatusWriter(edm::ParameterSet const& _ps) :
  channelStatus_(),
  towerStatus_(),
  firstRun_(_ps.getUntrackedParameter<unsigned>("firstRun"))
{
  std::ifstream inputFile(_ps.getUntrackedParameter<std::string>("inputFile"));
  if(!inputFile.is_open())
    throw cms::Exception("Invalid input for EcalDQMStatusWriter");

  ecaldqm::StatusManager statusManager;

  statusManager.readFromStream(inputFile);
  statusManager.writeToObj(channelStatus_, towerStatus_);
}

void
EcalDQMStatusWriter::analyze(edm::Event const&, edm::EventSetup const& _es)
{
  cond::service::PoolDBOutputService& dbOutput(*edm::Service<cond::service::PoolDBOutputService>());
  if(firstRun_ == dbOutput.endOfTime()) return; 

  dbOutput.writeOne(&channelStatus_, firstRun_, "EcalDQMChannelStatusRcd");
  dbOutput.writeOne(&towerStatus_, firstRun_, "EcalDQMTowerStatusRcd");

  firstRun_ = dbOutput.endOfTime(); // avoid accidentally re-writing the conditions
}

DEFINE_FWK_MODULE(EcalDQMStatusWriter);

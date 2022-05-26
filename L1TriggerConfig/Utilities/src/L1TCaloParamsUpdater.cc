#include <iomanip>
#include <fstream>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TCaloParamsUpdater : public edm::one::EDAnalyzer<> {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TCaloParamsUpdater(const edm::ParameterSet&) : token_{esConsumes()} {}

private:
  edm::ESGetToken<l1t::CaloParams, L1TCaloStage2ParamsRcd> token_;
};

void L1TCaloParamsUpdater::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  l1t::CaloParamsHelper m_params_helper(evSetup.getData(token_));

  //    std::ifstream is("tauL1CalibLUT_V2.txt");
  //    l1t::LUT lut;
  //    std::cout<<"LUT read success: "<<lut.read(is)<<std::endl;

  m_params_helper.setIsoTauEtaMax(28);

  std::shared_ptr<l1t::CaloParams> ptr1 = std::make_shared<l1t::CaloParams>(m_params_helper);

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(*ptr1, firstSinceTime, "L1TCaloStage2ParamsTweakedRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TCaloParamsUpdater);

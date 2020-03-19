#include <iomanip>
#include <fstream>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TCaloStage2ParamsRcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"
#include "L1Trigger/L1TCalorimeter/interface/CaloParamsHelper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TCaloParamsUpdater : public edm::EDAnalyzer {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TCaloParamsUpdater(const edm::ParameterSet&) : edm::EDAnalyzer() {}
  ~L1TCaloParamsUpdater(void) override {}
};

void L1TCaloParamsUpdater::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<l1t::CaloParams> handle1;
  //    evSetup.get<L1TCaloParamsRcd>().get( "l1conddb", handle1 ) ;
  evSetup.get<L1TCaloStage2ParamsRcd>().get(handle1);
  l1t::CaloParamsHelper m_params_helper(*(handle1.product()));

  //    std::ifstream is("tauL1CalibLUT_V2.txt");
  //    l1t::LUT lut;
  //    std::cout<<"LUT read success: "<<lut.read(is)<<std::endl;

  m_params_helper.setIsoTauEtaMax(28);

  std::shared_ptr<l1t::CaloParams> ptr1 = std::make_shared<l1t::CaloParams>(m_params_helper);

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOne(ptr1.get(), firstSinceTime, "L1TCaloStage2ParamsTweakedRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TCaloParamsUpdater);

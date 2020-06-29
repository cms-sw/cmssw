// -*- C++ -*-
//
// Package:    HcalSimulationConstantsESModule
// Class:      HcalSimulationConstantsESModule
//
/**\class HcalSimulationConstantsESModule Geometry/HcalCommonData/plugins/HcalSimulationConstantsESModule.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon Aug 12 16:40:29 PDT 2019
//
//

#include <memory>

#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/HcalCommonData/interface/HcalSimulationConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define EDM_ML_DEBUG

class HcalSimulationConstantsESModule : public edm::ESProducer {
public:
  HcalSimulationConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalSimulationConstants>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalSimNumberingRecord&);

private:
  edm::ESGetToken<HcalSimulationParameters, HcalParametersRcd> parSimToken_;
};

HcalSimulationConstantsESModule::HcalSimulationConstantsESModule(const edm::ParameterSet&) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "constructing HcalSimulationConstantsESModule";
#endif
  auto cc = setWhatProduced(this);
  parSimToken_ = cc.consumesFrom<HcalSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

void HcalSimulationConstantsESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hcalSimulationConstants", desc);
}

// ------------ method called to produce the data  ------------
HcalSimulationConstantsESModule::ReturnType HcalSimulationConstantsESModule::produce(
    const HcalSimNumberingRecord& iRecord) {
  const auto& parSim = iRecord.get(parSimToken_);
  return std::make_unique<HcalSimulationConstants>(&parSim);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalSimulationConstantsESModule);

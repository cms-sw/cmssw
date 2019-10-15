// -*- C++ -*-
//
// Package:    HcalDDDSimulationConstantsESModule
// Class:      HcalDDDSimulationConstantsESModule
//
/**\class HcalDDDSimulationConstantsESModule Geometry/HcalCommonData/plugins/HcalDDDSimulationConstantsESModule.cc

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

#include <Geometry/HcalCommonData/interface/HcalDDDSimulationConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define EDM_ML_DEBUG

class HcalDDDSimulationConstantsESModule : public edm::ESProducer {
public:
  HcalDDDSimulationConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalDDDSimulationConstants>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalSimNumberingRecord&);

private:
  edm::ESGetToken<HcalSimulationParameters, HcalParametersRcd> parSimToken_;
};

HcalDDDSimulationConstantsESModule::HcalDDDSimulationConstantsESModule(const edm::ParameterSet&) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "constructing HcalDDDSimulationConstantsESModule";
#endif
  auto cc = setWhatProduced(this);
  parSimToken_ = cc.consumesFrom<HcalSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

void HcalDDDSimulationConstantsESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hcalDDDSimulationConstants", desc);
}

// ------------ method called to produce the data  ------------
HcalDDDSimulationConstantsESModule::ReturnType HcalDDDSimulationConstantsESModule::produce(
    const HcalSimNumberingRecord& iRecord) {
  const auto& parSim = iRecord.get(parSimToken_);
  return std::make_unique<HcalDDDSimulationConstants>(&parSim);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDSimulationConstantsESModule);

// -*- C++ -*-
//
// Package:    CaloDDDSimulationConstantsESModule
// Class:      CaloDDDSimulationConstantsESModule
//
/**\class CaloDDDSimulationConstantsESModule Geometry/HcalCommonData/plugins/CaloDDDSimulationConstantsESModule.cc

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

#include <Geometry/HcalCommonData/interface/CaloDDDSimulationConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define EDM_ML_DEBUG

class CaloDDDSimulationConstantsESModule : public edm::ESProducer {
public:
  CaloDDDSimulationConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<CaloDDDSimulationConstants>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalSimNumberingRecord&);

private:
  edm::ESGetToken<CaloSimulationParameters, HcalParametersRcd> parSimToken_;
};

CaloDDDSimulationConstantsESModule::CaloDDDSimulationConstantsESModule(const edm::ParameterSet&) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "constructing CaloDDDSimulationConstantsESModule";
#endif
  auto cc = setWhatProduced(this);
  parSimToken_ = cc.consumesFrom<CaloSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

void CaloDDDSimulationConstantsESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("caloDDDSimulationConstants", desc);
}

// ------------ method called to produce the data  ------------
CaloDDDSimulationConstantsESModule::ReturnType CaloDDDSimulationConstantsESModule::produce(
    const HcalSimNumberingRecord& iRecord) {
  const auto& parSim = iRecord.get(parSimToken_);
  return std::make_unique<CaloDDDSimulationConstants>(&parSim);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CaloDDDSimulationConstantsESModule);

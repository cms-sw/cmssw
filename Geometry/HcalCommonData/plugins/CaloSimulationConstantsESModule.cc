// -*- C++ -*-
//
// Package:    CaloSimulationConstantsESModule
// Class:      CaloSimulationConstantsESModule
//
/**\class CaloSimulationConstantsESModule Geometry/HcalCommonData/plugins/CaloSimulationConstantsESModule.cc

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

#include <Geometry/HcalCommonData/interface/CaloSimulationConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define EDM_ML_DEBUG

class CaloSimulationConstantsESModule : public edm::ESProducer {
public:
  CaloSimulationConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<CaloSimulationConstants>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalSimNumberingRecord&);

private:
  edm::ESGetToken<CaloSimulationParameters, HcalParametersRcd> parSimToken_;
};

CaloSimulationConstantsESModule::CaloSimulationConstantsESModule(const edm::ParameterSet&) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "constructing CaloSimulationConstantsESModule";
#endif
  auto cc = setWhatProduced(this);
  parSimToken_ = cc.consumesFrom<CaloSimulationParameters, HcalParametersRcd>(edm::ESInputTag{});
}

void CaloSimulationConstantsESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("caloSimulationConstants", desc);
}

// ------------ method called to produce the data  ------------
CaloSimulationConstantsESModule::ReturnType CaloSimulationConstantsESModule::produce(
    const HcalSimNumberingRecord& iRecord) {
  const auto& parSim = iRecord.get(parSimToken_);
  return std::make_unique<CaloSimulationConstants>(&parSim);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(CaloSimulationConstantsESModule);

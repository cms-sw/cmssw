// -*- C++ -*-
//
// Package:    HcalDDDSimConstantsESModule
// Class:      HcalDDDSimConstantsESModule
//
/**\class HcalDDDSimConstantsESModule Geometry/HcalCommonData/plugins/HcalDDDSimConstantsESModule.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Dec 24 16:40:29 PDT 2013
//
//

#include <memory>

#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/HcalCommonData/interface/HcalDDDSimConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define EDM_ML_DEBUG

class HcalDDDSimConstantsESModule : public edm::ESProducer {
public:
  HcalDDDSimConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalDDDSimConstants>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const HcalSimNumberingRecord&);

private:
  edm::ESGetToken<HcalParameters, HcalParametersRcd> parToken_;
};

HcalDDDSimConstantsESModule::HcalDDDSimConstantsESModule(const edm::ParameterSet&) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "constructing HcalDDDSimConstantsESModule";
#endif
  auto cc = setWhatProduced(this);
  parToken_ = cc.consumesFrom<HcalParameters, HcalParametersRcd>(edm::ESInputTag{});
}

void HcalDDDSimConstantsESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hcalDDDSimConstants", desc);
}

// ------------ method called to produce the data  ------------
HcalDDDSimConstantsESModule::ReturnType HcalDDDSimConstantsESModule::produce(const HcalSimNumberingRecord& iRecord) {
  const auto& par = iRecord.get(parToken_);
  return std::make_unique<HcalDDDSimConstants>(&par);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDSimConstantsESModule);

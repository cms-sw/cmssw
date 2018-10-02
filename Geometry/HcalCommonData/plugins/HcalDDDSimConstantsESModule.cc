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
#include <FWCore/Framework/interface/ESHandle.h>

#include <Geometry/HcalCommonData/interface/HcalDDDSimConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

class HcalDDDSimConstantsESModule : public edm::ESProducer {

public:
  HcalDDDSimConstantsESModule(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<HcalDDDSimConstants>;

  static void fillDescriptions( edm::ConfigurationDescriptions & );

  ReturnType produce(const HcalSimNumberingRecord&);
};

HcalDDDSimConstantsESModule::HcalDDDSimConstantsESModule(const edm::ParameterSet&) {
  setWhatProduced(this);
}

void HcalDDDSimConstantsESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  edm::ParameterSetDescription desc;
  descriptions.add( "hcalDDDSimConstants", desc );
}

// ------------ method called to produce the data  ------------
HcalDDDSimConstantsESModule::ReturnType
HcalDDDSimConstantsESModule::produce(const HcalSimNumberingRecord& iRecord) {

  const HcalParametersRcd& parRecord = iRecord.getRecord<HcalParametersRcd>();
  edm::ESHandle<HcalParameters> parHandle;
  parRecord.get(parHandle);

  return std::make_unique<HcalDDDSimConstants>(parHandle.product());
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDSimConstantsESModule);

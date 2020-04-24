// -*- C++ -*-
//
// Package:    HcalDDDRecConstantsESModule
// Class:      HcalDDDRecConstantsESModule
// 
/**\class HcalDDDRecConstantsESModule HcalDDDRecConstantsESModule.h Geometry/HcalCommonData/interface/HcalDDDRecConstantsESModule.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Dec 24 16:40:29 PDT 2013
// $Id: HcalDDDRecConstantsESModule.cc,v 1.0 2013/12/24 12:47:41 sunanda Exp $
//
//


// system include files
#include <memory>

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <CondFormats/GeometryObjects/interface/HcalParameters.h>
#include <Geometry/HcalCommonData/interface/HcalDDDSimConstants.h>
#include <Geometry/HcalCommonData/interface/HcalDDDRecConstants.h>
#include <Geometry/Records/interface/HcalRecNumberingRecord.h>

//#define EDM_ML_DEBUG

class HcalDDDRecConstantsESModule : public edm::ESProducer {

public:
  HcalDDDRecConstantsESModule(const edm::ParameterSet&);
  ~HcalDDDRecConstantsESModule() override;

  typedef std::shared_ptr<HcalDDDRecConstants> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & );

  ReturnType produce(const HcalRecNumberingRecord&);

private:
  HcalDDDRecConstants* hcalDDDConst_;
};

HcalDDDRecConstantsESModule::HcalDDDRecConstantsESModule(const edm::ParameterSet& iConfig) : hcalDDDConst_(nullptr) {
#ifdef EDM_ML_DEBUG
  std::cout <<"constructing HcalDDDRecConstantsESModule" << std::endl;
#endif
  setWhatProduced(this);
}

HcalDDDRecConstantsESModule::~HcalDDDRecConstantsESModule() {}

void HcalDDDRecConstantsESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  edm::ParameterSetDescription desc;
  descriptions.add( "hcalDDDRecConstants", desc );
}

// ------------ method called to produce the data  ------------
HcalDDDRecConstantsESModule::ReturnType
HcalDDDRecConstantsESModule::produce(const HcalRecNumberingRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  std::cout << "in HcalDDDRecConstantsESModule::produce" << std::endl;
#endif
  if (hcalDDDConst_ == nullptr) {
    edm::ESHandle<HcalParameters>         parHandle;
    iRecord.getRecord<HcalParametersRcd>().get(parHandle);
    edm::ESHandle<HcalDDDSimConstants>    hdc;
    iRecord.getRecord<HcalSimNumberingRecord>().get(hdc);
    hcalDDDConst_ = new HcalDDDRecConstants(&(*parHandle), *hdc);
  }
  return HcalDDDRecConstantsESModule::ReturnType(hcalDDDConst_) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDRecConstantsESModule);

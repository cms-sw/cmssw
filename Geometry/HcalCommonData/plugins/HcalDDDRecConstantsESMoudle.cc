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

  using ReturnType = std::unique_ptr<HcalDDDRecConstants>;

  static void fillDescriptions( edm::ConfigurationDescriptions & );

  ReturnType produce(const HcalRecNumberingRecord&);

private:
  edm::ESGetToken<HcalParameters, HcalParametersRcd> parToken_;
  edm::ESGetToken<HcalDDDSimConstants, HcalSimNumberingRecord> hdcToken_;
};

HcalDDDRecConstantsESModule::HcalDDDRecConstantsESModule(const edm::ParameterSet& iConfig) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") <<"constructing HcalDDDRecConstantsESModule";
#endif
  auto cc = setWhatProduced(this);
  parToken_ = cc.consumesFrom<HcalParameters, HcalParametersRcd>(edm::ESInputTag{});
  hdcToken_ = cc.consumesFrom<HcalDDDSimConstants, HcalSimNumberingRecord>(edm::ESInputTag{});
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
  edm::LogVerbatim("HcalGeom") << "in HcalDDDRecConstantsESModule::produce";
#endif
  const auto& par = iRecord.get(parToken_);
  const auto& hdc = iRecord.get(hdcToken_);

  return std::make_unique<HcalDDDRecConstants>(&par, hdc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDRecConstantsESModule);

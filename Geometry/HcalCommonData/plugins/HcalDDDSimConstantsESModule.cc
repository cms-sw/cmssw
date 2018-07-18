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

// system include files
#include <memory>

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <Geometry/HcalCommonData/interface/HcalDDDSimConstants.h>
#include <Geometry/Records/interface/HcalSimNumberingRecord.h>

//#define EDM_ML_DEBUG

class HcalDDDSimConstantsESModule : public edm::ESProducer {

public:
  HcalDDDSimConstantsESModule(const edm::ParameterSet&);
  ~HcalDDDSimConstantsESModule() override;

  typedef std::shared_ptr<HcalDDDSimConstants> ReturnType;

  static void fillDescriptions( edm::ConfigurationDescriptions & );

  ReturnType produce(const HcalSimNumberingRecord&);

  void initializeHcalDDDSimConstants( const HcalParametersRcd& igr);

private:
  HcalDDDSimConstants* hcalDDDConst_;
};

HcalDDDSimConstantsESModule::HcalDDDSimConstantsESModule(const edm::ParameterSet& iConfig) : hcalDDDConst_(nullptr) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") <<"constructing HcalDDDSimConstantsESModule";
#endif
  setWhatProduced(this, dependsOn(&HcalDDDSimConstantsESModule::initializeHcalDDDSimConstants));
}

HcalDDDSimConstantsESModule::~HcalDDDSimConstantsESModule() {}

void HcalDDDSimConstantsESModule::fillDescriptions( edm::ConfigurationDescriptions & descriptions ) {
  edm::ParameterSetDescription desc;
  descriptions.add( "hcalDDDSimConstants", desc );
}

// ------------ method called to produce the data  ------------
HcalDDDSimConstantsESModule::ReturnType
HcalDDDSimConstantsESModule::produce(const HcalSimNumberingRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "in HcalDDDSimConstantsESModule::produce";
#endif
  if (hcalDDDConst_ == nullptr) {
    edm::LogError("HCalGeom") << "HcalDDDSimConstantsESModule::produceHcalDDDSimConstants has NOT been initialized!";
    throw cms::Exception("DDException") << "HcalDDDSimConstantsESModule::Cannot produce HcalDDDSimConstnats";
  }
  return HcalDDDSimConstantsESModule::ReturnType(hcalDDDConst_) ;
}

void HcalDDDSimConstantsESModule::initializeHcalDDDSimConstants(const HcalParametersRcd& igr) {

  std::string                   label_;
  edm::ESHandle<HcalParameters> parHandle;
  igr.get(label_, parHandle);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "in HcalDDDSimConstantsESModule::initializeHcalDDDSimConstants";
#endif
  const HcalParameters* hpar = &(*parHandle);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HcalGeom") << "about to make my new hcalDDDConst_ with " 
			       << hpar;
#endif
  hcalDDDConst_ = new HcalDDDSimConstants(hpar);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HcalDDDSimConstantsESModule);

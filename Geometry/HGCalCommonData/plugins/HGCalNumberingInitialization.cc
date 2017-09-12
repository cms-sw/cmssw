// -*- C++ -*-
//
// Package:    HGCalNumberingInitialization
// Class:      HGCalNumberingInitialization
// 
/**\class HGCalNumberingInitialization HGCalNumberingInitialization.h Geometry/HGCalCommonData/interface/HGCalNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Tue Mar 21 16:40:29 PDT 2013
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#define EDM_ML_DEBUG

class HGCalNumberingInitialization : public edm::ESProducer {

public:
  HGCalNumberingInitialization(const edm::ParameterSet&);
  ~HGCalNumberingInitialization() override;

  typedef std::unique_ptr<HGCalDDDConstants> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  HGCalDDDConstants* hgcalDDDConst_;
  std::string        name_;
};

HGCalNumberingInitialization::HGCalNumberingInitialization(const edm::ParameterSet& iConfig) : hgcalDDDConst_(nullptr) {

  name_ = iConfig.getUntrackedParameter<std::string>("Name");
  edm::LogInfo("HGCalGeom") << "HGCalNumberingInitialization for " << name_;
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalNumberingInitialization for " << name_ << std::endl;
#endif
  setWhatProduced(this, name_);
}

HGCalNumberingInitialization::~HGCalNumberingInitialization() {}


// ------------ method called to produce the data  ------------
HGCalNumberingInitialization::ReturnType
HGCalNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {

  edm::LogInfo("HGCalGeom") << "in HGCalNumberingInitialization::produce";
  if (hgcalDDDConst_ == nullptr) {
    edm::ESHandle<HGCalParameters>  pHGpar;
    iRecord.get(name_, pHGpar);
    hgcalDDDConst_ = new HGCalDDDConstants(&(*pHGpar), name_);
  }
  return ReturnType(hgcalDDDConst_) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalNumberingInitialization);


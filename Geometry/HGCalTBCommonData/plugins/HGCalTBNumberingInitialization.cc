// -*- C++ -*-
//
// Package:    HGCalNumberingInitialization
// Class:      HGCalNumberingInitialization
//
/**\class HGCalNumberingInitialization HGCalNumberingInitialization.h
 Geometry/HGCalCommonData/interface/HGCalNumberingInitialization.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Fri Dec 30 16:40:29 PDT 2022
//

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalTBNumberingInitialization : public edm::ESProducer {
public:
  HGCalTBNumberingInitialization(const edm::ParameterSet&);
  ~HGCalTBNumberingInitialization() override = default;

  typedef std::unique_ptr<HGCalTBDDDConstants> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<HGCalTBParameters, IdealGeometryRecord> hgParToken_;
  std::string name_;
};

HGCalTBNumberingInitialization::HGCalTBNumberingInitialization(const edm::ParameterSet& iConfig) {
  name_ = iConfig.getUntrackedParameter<std::string>("name");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBNumberingInitialization for " << name_;
#endif
  auto cc = setWhatProduced(this, name_);
  hgParToken_ = cc.consumes<HGCalTBParameters>(edm::ESInputTag{"", name_});
}

// ------------ method called to produce the data  ------------
HGCalTBNumberingInitialization::ReturnType HGCalTBNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {
  edm::LogVerbatim("HGCalGeom") << "in HGCalTBNumberingInitialization::produce for " << name_;
  const auto& pHGpar = iRecord.get(hgParToken_);
  return std::make_unique<HGCalTBDDDConstants>(&pHGpar, name_);
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalTBNumberingInitialization);

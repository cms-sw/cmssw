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
//         Created:  Tue Mar 21 16:40:29 PDT 2013
//

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#define EDM_ML_DEBUG

class HGCalNumberingInitialization : public edm::ESProducer {
public:
  HGCalNumberingInitialization(const edm::ParameterSet&);
  ~HGCalNumberingInitialization() override;

  typedef std::unique_ptr<HGCalDDDConstants> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<HGCalParameters, IdealGeometryRecord> hgParToken_;
  std::string name_;
};

HGCalNumberingInitialization::HGCalNumberingInitialization(const edm::ParameterSet& iConfig) {
  name_ = iConfig.getUntrackedParameter<std::string>("Name");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalNumberingInitialization for " << name_;
#endif
  auto cc = setWhatProduced(this, name_);
  hgParToken_ = cc.consumes<HGCalParameters>(edm::ESInputTag{"", name_});
}

HGCalNumberingInitialization::~HGCalNumberingInitialization() {}

// ------------ method called to produce the data  ------------
HGCalNumberingInitialization::ReturnType HGCalNumberingInitialization::produce(const IdealGeometryRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "in HGCalNumberingInitialization::produce for " << name_;
#endif
  const auto& pHGpar = iRecord.get(hgParToken_);
  return std::make_unique<HGCalDDDConstants>(&pHGpar, name_);
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalNumberingInitialization);

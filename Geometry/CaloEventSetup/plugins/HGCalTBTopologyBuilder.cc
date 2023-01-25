// -*- C++ -*-
//
// Package:    CaloEventSetup
// Class:      HGCalTBTopologyBuilder
//
/**\class HGCalTBTopologyBuilder HGCalTBTopologyBuilder.h 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//
//

// system include files
#include <memory>

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "Geometry/CaloTopology/interface/HGCalTBTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class HGCalTBTopologyBuilder : public edm::ESProducer {
public:
  HGCalTBTopologyBuilder(const edm::ParameterSet& iP);
  ~HGCalTBTopologyBuilder() override;

  using ReturnType = std::unique_ptr<HGCalTBTopology>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<HGCalTBDDDConstants, IdealGeometryRecord> hgcToken_;
  int det_;
};

HGCalTBTopologyBuilder::HGCalTBTopologyBuilder(const edm::ParameterSet& iConfig) {
  auto name = iConfig.getParameter<std::string>("Name");
  det_ = iConfig.getParameter<int>("Type");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "constructing HGCalTBTopology for " << name << " and det " << det_;
#endif
  hgcToken_ = setWhatProduced(this, name).consumes<HGCalTBDDDConstants>(edm::ESInputTag{"", name});
}

HGCalTBTopologyBuilder::~HGCalTBTopologyBuilder() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
HGCalTBTopologyBuilder::ReturnType HGCalTBTopologyBuilder::produce(const IdealGeometryRecord& iRecord) {
  const HGCalTBDDDConstants& hgdc = iRecord.get(hgcToken_);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create HGCalTBTopology(hgdc,det)";
#endif
  return std::make_unique<HGCalTBTopology>(&hgdc, det_);
}

DEFINE_FWK_EVENTSETUP_MODULE(HGCalTBTopologyBuilder);

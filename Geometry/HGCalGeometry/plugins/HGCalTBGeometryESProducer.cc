// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalTBGeometryESProducer
//
/**\class HGCalTBGeometryESProducer HGCalTBGeometryESProducer.h 

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
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTBTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometryLoader.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class HGCalTBGeometryESProducer : public edm::ESProducer {
public:
  HGCalTBGeometryESProducer(const edm::ParameterSet& iP);
  ~HGCalTBGeometryESProducer() override;

  using ReturnType = std::unique_ptr<HGCalTBGeometry>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<HGCalTBTopology, IdealGeometryRecord> topologyToken_;
  std::string name_;
};

HGCalTBGeometryESProducer::HGCalTBGeometryESProducer(const edm::ParameterSet& iConfig) {
  name_ = iConfig.getUntrackedParameter<std::string>("Name");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Constructing HGCalTBGeometry for " << name_;
#endif
  auto cc = setWhatProduced(this, name_);
  topologyToken_ = cc.consumes<HGCalTBTopology>(edm::ESInputTag{"", name_});
}

HGCalTBGeometryESProducer::~HGCalTBGeometryESProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
HGCalTBGeometryESProducer::ReturnType HGCalTBGeometryESProducer::produce(const IdealGeometryRecord& iRecord) {
  const auto& topo = iRecord.get(topologyToken_);
  edm::LogVerbatim("HGCalGeom") << "Create HGCalTBGeometry (*topo) for " << name_;

  HGCalTBGeometryLoader builder;
  return ReturnType(builder.build(topo));
}

DEFINE_FWK_EVENTSETUP_MODULE(HGCalTBGeometryESProducer);

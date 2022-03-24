// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      FastTimeGeometryESProducer
//
/**\class FastTimeGeometryESProducer FastTimeGeometryESProducer.h 

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
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometryLoader.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class FastTimeGeometryESProducer : public edm::ESProducer {
public:
  FastTimeGeometryESProducer(const edm::ParameterSet& iP);
  ~FastTimeGeometryESProducer() override = default;

  using ReturnType = std::unique_ptr<FastTimeGeometry>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<FastTimeTopology, IdealGeometryRecord> topologyToken_;
};

FastTimeGeometryESProducer::FastTimeGeometryESProducer(const edm::ParameterSet& iConfig) {
  auto name = iConfig.getUntrackedParameter<std::string>("Name");
#ifdef EDM_ML_DEBUG
  auto type = iConfig.getUntrackedParameter<int>("Type");
  edm::LogVerbatim("HGCalGeom") << "constructing FastTimeGeometry for " << name << " Type " << type;
#endif
  auto cc = setWhatProduced(this, name);
  topologyToken_ = cc.consumes<FastTimeTopology>(edm::ESInputTag{"", name});
}

//
// member functions
//

// ------------ method called to produce the data  ------------
FastTimeGeometryESProducer::ReturnType FastTimeGeometryESProducer::produce(const IdealGeometryRecord& iRecord) {
  const auto& topo = iRecord.get(topologyToken_);

  FastTimeGeometryLoader builder;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create FastTimeGeometry (topo)";
#endif
  return ReturnType(builder.build(topo));
}

DEFINE_FWK_EVENTSETUP_MODULE(FastTimeGeometryESProducer);

// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalGeometryESProducer
// 
/**\class HGCalGeometryESProducer HGCalGeometryESProducer.h 

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
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometryLoader.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class HGCalGeometryESProducer : public edm::ESProducer {

public:
  HGCalGeometryESProducer( const edm::ParameterSet& iP );
  ~HGCalGeometryESProducer() override ;

  using ReturnType = std::unique_ptr<HGCalGeometry>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<HGCalTopology, IdealGeometryRecord> topologyToken_;
};


HGCalGeometryESProducer::HGCalGeometryESProducer(const edm::ParameterSet& iConfig) {

  auto name     = iConfig.getUntrackedParameter<std::string>("Name");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Constructing HGCalGeometry for " << name;
#endif
  auto cc = setWhatProduced(this, name);
  topologyToken_ = cc.consumes<HGCalTopology>(edm::ESInputTag{"", name});
}


HGCalGeometryESProducer::~HGCalGeometryESProducer() { }


//
// member functions
//

// ------------ method called to produce the data  ------------
HGCalGeometryESProducer::ReturnType
HGCalGeometryESProducer::produce(const IdealGeometryRecord& iRecord ) {

  const auto& topo = iRecord.get(topologyToken_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create HGCalGeometry (*topo)";
#endif

  HGCalGeometryLoader builder;
  return ReturnType(builder.build(topo));
}

DEFINE_FWK_EVENTSETUP_MODULE(HGCalGeometryESProducer);

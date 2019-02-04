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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometryLoader.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class FastTimeGeometryESProducer : public edm::ESProducer {

public:
  FastTimeGeometryESProducer( const edm::ParameterSet& iP );
  ~FastTimeGeometryESProducer() override ;

  using ReturnType = std::unique_ptr<FastTimeGeometry>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  std::string        name_;
  int                type_;
};


FastTimeGeometryESProducer::FastTimeGeometryESProducer(const edm::ParameterSet& iConfig) {

  name_     = iConfig.getUntrackedParameter<std::string>("Name");
  type_     = iConfig.getUntrackedParameter<int>("Type");
#ifdef EDM_ML_DEBUG
  std::cout <<"constructing FastTimeGeometry for " << name_ << " Type "
	    << type_ << std::endl;
#endif
  setWhatProduced(this, name_);
}


FastTimeGeometryESProducer::~FastTimeGeometryESProducer() { }


//
// member functions
//

// ------------ method called to produce the data  ------------
FastTimeGeometryESProducer::ReturnType
FastTimeGeometryESProducer::produce(const IdealGeometryRecord& iRecord ) {

  edm::ESHandle<FastTimeTopology> topo;
  iRecord.get(name_,topo);

  FastTimeGeometryLoader builder;
#ifdef EDM_ML_DEBUG
  std::cout << "Create FastTimeGeometry (*topo)" << std::endl;
#endif
  return ReturnType(builder.build(*topo));
}

DEFINE_FWK_EVENTSETUP_MODULE(FastTimeGeometryESProducer);

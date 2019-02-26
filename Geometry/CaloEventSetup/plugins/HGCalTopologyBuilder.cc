// -*- C++ -*-
//
// Package:    CaloEventSetup
// Class:      HGCalTopologyBuilder
// 
/**\class HGCalTopologyBuilder HGCalTopologyBuilder.h 

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

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class HGCalTopologyBuilder : public edm::ESProducer {

public:
  HGCalTopologyBuilder( const edm::ParameterSet& iP );
  ~HGCalTopologyBuilder() override ;

  using ReturnType = std::unique_ptr<HGCalTopology>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  std::string        name_;
  int                det_;
};


HGCalTopologyBuilder::HGCalTopologyBuilder(const edm::ParameterSet& iConfig) {

  name_     = iConfig.getParameter<std::string>("Name");
  det_      = iConfig.getParameter<int>("Type");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "constructing HGCalTopology for " << name_
				<< " and det " << det_;
#endif
  setWhatProduced(this, name_);
}


HGCalTopologyBuilder::~HGCalTopologyBuilder() { }

//
// member functions
//

// ------------ method called to produce the data  ------------
HGCalTopologyBuilder::ReturnType
HGCalTopologyBuilder::produce(const IdealGeometryRecord& iRecord ) {

  edm::ESHandle<HGCalDDDConstants>  pHGDC;
  iRecord.get(name_, pHGDC);
  const HGCalDDDConstants & hgdc = (*pHGDC);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create HGCalTopology(hgdc,det)";
#endif
  return std::make_unique<HGCalTopology>(hgdc, det_);
}

DEFINE_FWK_EVENTSETUP_MODULE(HGCalTopologyBuilder);

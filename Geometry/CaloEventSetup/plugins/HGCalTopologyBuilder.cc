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
#include "boost/shared_ptr.hpp"

// user include files
#include <FWCore/Framework/interface/ModuleFactory.h>
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define DebugLog

//
// class decleration
//

class HGCalTopologyBuilder : public edm::ESProducer {

public:
  HGCalTopologyBuilder( const edm::ParameterSet& iP );
  ~HGCalTopologyBuilder() ;

  typedef boost::shared_ptr< HGCalTopology > ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  std::string        name_;
  bool               halfType_;
  ForwardSubdetector subdet_;
};


HGCalTopologyBuilder::HGCalTopologyBuilder(const edm::ParameterSet& iConfig) {

  name_     = iConfig.getUntrackedParameter<std::string>("Name");
  int type  = iConfig.getUntrackedParameter<int>("Type");
  halfType_ = iConfig.getUntrackedParameter<bool>("HalfType");
  if (type == 0) {
    subdet_ = HGCEE;
  } else if (type == 1) {
    subdet_ = HGCHEF;
  } else if (type == 2) {
    subdet_ = HGCHEB;
  } else {
    subdet_ = HGCHET;
  }
#ifdef DebugLog
  std::cout <<"constructing HGCalTopology for " << name_ << std::endl;
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
  iRecord.get(name_, pHGDC) ;
  const HGCalDDDConstants & hgdc = (*pHGDC);

  ReturnType ct ( new HGCalTopology(hgdc, subdet_, halfType_) ) ;
#ifdef DebugLog
  std::cout << "Create HGCalTopology(hgdc,subdet,type)" << std::endl;
#endif
  return ct ;
}

DEFINE_FWK_EVENTSETUP_MODULE(HGCalTopologyBuilder);

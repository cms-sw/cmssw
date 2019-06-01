// -*- C++ -*-
//
// Package:    CaloEventSetup
// Class:      FastTimeTopologyBuilder
//
/**\class FastTimeTopologyBuilder FastTimeTopologyBuilder.h 

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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

//#define EDM_ML_DEBUG

//
// class decleration
//

class FastTimeTopologyBuilder : public edm::ESProducer {
public:
  FastTimeTopologyBuilder(const edm::ParameterSet& iP);
  ~FastTimeTopologyBuilder() override;

  using ReturnType = std::unique_ptr<FastTimeTopology>;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<FastTimeDDDConstants, IdealGeometryRecord> ftlToken_;
  int type_;
  ForwardSubdetector subdet_;
};

FastTimeTopologyBuilder::FastTimeTopologyBuilder(const edm::ParameterSet& iConfig) {
  auto name = iConfig.getUntrackedParameter<std::string>("Name");
  type_ = iConfig.getUntrackedParameter<int>("Type");
  subdet_ = FastTime;
#ifdef EDM_ML_DEBUG
  std::cout << "constructing FastTimeTopology for " << name << " Type " << type_ << std::endl;
#endif
  ftlToken_ = setWhatProduced(this, name).consumes<FastTimeDDDConstants>(edm::ESInputTag{"", name});
}

FastTimeTopologyBuilder::~FastTimeTopologyBuilder() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
FastTimeTopologyBuilder::ReturnType FastTimeTopologyBuilder::produce(const IdealGeometryRecord& iRecord) {
  const FastTimeDDDConstants& hgdc = iRecord.get(ftlToken_);

#ifdef EDM_ML_DEBUG
  std::cout << "Create FastTimeTopology(hgdc,subdet,type)" << std::endl;
#endif
  return std::make_unique<FastTimeTopology>(hgdc, subdet_, type_);
}

DEFINE_FWK_EVENTSETUP_MODULE(FastTimeTopologyBuilder);

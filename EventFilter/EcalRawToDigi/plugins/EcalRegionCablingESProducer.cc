// -*- C++ -*-
//
// Package:    EcalRegionCablingESProducer
// Class:      EcalRegionCablingESProducer
//
/**\class EcalRegionCablingESProducer EcalRegionCablingESProducer.h EventFilter/EcalRegionCablingESProducer/src/EcalRegionCablingESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Sun Oct  7 00:37:06 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "EventFilter/EcalRawToDigi/interface/EcalRegionCablingRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRegionCabling.h"

class EcalRegionCablingESProducer : public edm::ESProducer {
public:
  EcalRegionCablingESProducer(const edm::ParameterSet&);
  ~EcalRegionCablingESProducer() override;

  typedef std::unique_ptr<EcalRegionCabling> ReturnType;

  ReturnType produce(const EcalRegionCablingRecord&);

private:
  edm::ParameterSet conf_;

  edm::ESGetToken<EcalElectronicsMapping, EcalMappingRcd> esEcalElectronicsMappingToken_;
};

EcalRegionCablingESProducer::EcalRegionCablingESProducer(const edm::ParameterSet& iConfig) : conf_(iConfig) {
  auto cc = setWhatProduced(this);
  esEcalElectronicsMappingToken_ = cc.consumesFrom<EcalElectronicsMapping, EcalMappingRcd>();
}

EcalRegionCablingESProducer::~EcalRegionCablingESProducer() {}

EcalRegionCablingESProducer::ReturnType EcalRegionCablingESProducer::produce(const EcalRegionCablingRecord& iRecord) {
  edm::ESHandle<EcalElectronicsMapping> mapping = iRecord.getHandle(esEcalElectronicsMappingToken_);

  return std::make_unique<EcalRegionCabling>(conf_, mapping.product());
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(EcalRegionCablingESProducer);

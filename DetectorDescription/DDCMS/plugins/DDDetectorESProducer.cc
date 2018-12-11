// -*- C++ -*-
//
// Package:    DetectorDescription/DDDetectorESProducer
// Class:      DDDetectorESProducer
// 
/**\class DDDetectorESProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Ianna Osborne
//         Created:  Fri, 07 Dec 2018 11:20:52 GMT
//
//

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

class DDDetectorESProducer : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
public:
  DDDetectorESProducer(const edm::ParameterSet&);
  ~DDDetectorESProducer() override;
  
  using ReturnType = std::unique_ptr<cms::DDDetector>;
  
  ReturnType produce(const DetectorDescriptionRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, const edm::IOVSyncValue&, edm::ValidityInterval&) override;
  
private:
  std::string m_confGeomXMLFiles;
};

DDDetectorESProducer::DDDetectorESProducer(const edm::ParameterSet& iConfig)
  : m_confGeomXMLFiles(iConfig.getParameter<std::string>("confGeomXMLFiles"))
{
   setWhatProduced(this);
   findingRecord<DetectorDescriptionRcd>();
}

DDDetectorESProducer::~DDDetectorESProducer()
{
}

void
DDDetectorESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;

  desc.add<std::string>("confGeomXMLFiles");
  descriptions.addDefault(desc);
}

void
DDDetectorESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

DDDetectorESProducer::ReturnType
DDDetectorESProducer::produce(const DetectorDescriptionRcd& iRecord)
{
  auto product = std::make_unique<cms::DDDetector>();
  product->process(m_confGeomXMLFiles);
  return product;
}

#include "FWCore/Framework/interface/SourceFactory.h"
DEFINE_FWK_EVENTSETUP_SOURCE(DDDetectorESProducer);

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

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

using namespace cms;
using namespace std;

class DDDetectorESProducer : public edm::ESProducer,
			     public edm::EventSetupRecordIntervalFinder {
public:
  DDDetectorESProducer(const edm::ParameterSet&);
  ~DDDetectorESProducer() override;
  
  using ReturnType = unique_ptr<cms::DDDetector>;
  
  ReturnType produce(const DetectorDescriptionRcd&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
		      const edm::IOVSyncValue&, edm::ValidityInterval&) override;
  
private:
  string m_confGeomXMLFiles;
  string m_label;
};

DDDetectorESProducer::DDDetectorESProducer(const edm::ParameterSet& iConfig)
  : m_confGeomXMLFiles(iConfig.getParameter<edm::FileInPath>("confGeomXMLFiles").fullPath()),
    m_label(iConfig.getParameter<std::string>("label"))
{
  setWhatProduced(this, m_label);
  findingRecord<DetectorDescriptionRcd>();
}

DDDetectorESProducer::~DDDetectorESProducer()
{
}

void
DDDetectorESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;

  desc.add<edm::FileInPath>("confGeomXMLFiles");
  desc.add<string>("label");
  descriptions.addDefault(desc);
}

void
DDDetectorESProducer::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
				     const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval) {
  oInterval = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(), edm::IOVSyncValue::endOfTime()); //infinite
}

DDDetectorESProducer::ReturnType
DDDetectorESProducer::produce(const DetectorDescriptionRcd& iRecord)
{
  cout << "DDDetectorESProducer::Produce " << m_label << "\n";
  using Detector = dd4hep::Detector;
  auto product = make_unique<DDDetector>();
  
  product->description = &Detector::getInstance(m_label);
  product->description->addExtension<DDVectorsMap>(&product->vectors);
  product->description->addExtension<DDPartSelectionMap>(&product->partsels);
  product->description->addExtension<DDSpecParRegistry>(&product->specpars);
  
  string name("DD4hep_CompactLoader");
  const char* files[] = { m_confGeomXMLFiles.c_str(), nullptr };
  product->description->apply(name.c_str(), 2, (char**)files);

  return product;
}

DEFINE_FWK_EVENTSETUP_SOURCE(DDDetectorESProducer);

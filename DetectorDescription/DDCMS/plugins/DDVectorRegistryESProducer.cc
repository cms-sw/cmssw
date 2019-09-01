// -*- C++ -*-
//
// Package:    DetectorDescription/DDCMS
// Class:      DDVectorRegistryESProducer
//
/**\class DDVectorRegistryESProducer

 Description: Produce Vector registry

 Implementation:
     Vectors are defined in XML
*/
//
// Original Author:  Ianna Osborne
//         Created:  Fri, 07 Dec 2018 11:20:52 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

using namespace std;
using namespace cms;
using namespace edm;

class DDVectorRegistryESProducer : public edm::ESProducer {
public:
  DDVectorRegistryESProducer(const edm::ParameterSet&);
  ~DDVectorRegistryESProducer() override;

  using ReturnType = unique_ptr<DDVectorRegistry>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const DDVectorRegistryRcd&);

private:
  const string m_label;
};

DDVectorRegistryESProducer::DDVectorRegistryESProducer(const edm::ParameterSet& iConfig)
    : m_label(iConfig.getParameter<string>("appendToDataLabel")) {
  setWhatProduced(this);
}

DDVectorRegistryESProducer::~DDVectorRegistryESProducer() {}

void DDVectorRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDVectorRegistryESProducer::ReturnType DDVectorRegistryESProducer::produce(const DDVectorRegistryRcd& iRecord) {
  LogDebug("Geometry") << "DDVectorRegistryESProducer::produce\n";
  edm::ESHandle<DDDetector> det;
  iRecord.getRecord<GeometryFileRcd>().get(m_label, det);

  const DDVectorsMap& registry = det->vectors();

  auto product = std::make_unique<DDVectorRegistry>();
  product->vectors.insert(registry.begin(), registry.end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDVectorRegistryESProducer);

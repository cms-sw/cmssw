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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

#include <unordered_map>
#include <vector>

using namespace std;
using namespace cms;
using namespace edm;

using DDVectorsMap = std::unordered_map<std::string, std::vector<double>>;

class DDVectorRegistryESProducer : public edm::ESProducer {
public:
  DDVectorRegistryESProducer(const edm::ParameterSet&);
  ~DDVectorRegistryESProducer() override;

  using ReturnType = unique_ptr<DDVectorRegistry>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const DDVectorRegistryRcd&);

private:
  const edm::ESGetToken<DDDetector, IdealGeometryRecord> m_token;
};

DDVectorRegistryESProducer::DDVectorRegistryESProducer(const edm::ParameterSet& iConfig)
    : m_token(setWhatProduced(this).consumes(edm::ESInputTag("", iConfig.getParameter<string>("appendToDataLabel")))) {}

DDVectorRegistryESProducer::~DDVectorRegistryESProducer() {}

void DDVectorRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDVectorRegistryESProducer::ReturnType DDVectorRegistryESProducer::produce(const DDVectorRegistryRcd& iRecord) {
  LogDebug("Geometry") << "DDVectorRegistryESProducer::produce\n";
  const auto& registry = iRecord.get(m_token).vectors();

  auto product = std::make_unique<DDVectorRegistry>();
  product->vectors.insert(registry.begin(), registry.end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDVectorRegistryESProducer);

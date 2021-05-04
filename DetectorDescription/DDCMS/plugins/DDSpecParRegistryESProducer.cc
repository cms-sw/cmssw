// -*- C++ -*-
//
// Package:    DetectorDescription/DDCMS
// Class:      DDSpecParRegistryESProducer
//
/**\class DDSpecParRegistryESProducer

 Description: Produce SpecPar registry

 Implementation:
     SpecPars are described in XML
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 09 Jan 2019 16:04:31 GMT
//
//

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "Geometry/Records/interface/DDSpecParRegistryRcd.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include <DD4hep/Detector.h>
#include <DD4hep/SpecParRegistry.h>

using namespace std;
using namespace cms;

class DDSpecParRegistryESProducer : public edm::ESProducer {
public:
  DDSpecParRegistryESProducer(const edm::ParameterSet&);
  ~DDSpecParRegistryESProducer() override;

  using ReturnType = unique_ptr<dd4hep::SpecParRegistry>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const DDSpecParRegistryRcd&);

private:
  const edm::ESGetToken<DDDetector, IdealGeometryRecord> m_token;
};

DDSpecParRegistryESProducer::DDSpecParRegistryESProducer(const edm::ParameterSet& iConfig)
    : m_token(
          setWhatProduced(this).consumes(edm::ESInputTag("", iConfig.getParameter<std::string>("appendToDataLabel")))) {
}

DDSpecParRegistryESProducer::~DDSpecParRegistryESProducer() {}

void DDSpecParRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDSpecParRegistryESProducer::ReturnType DDSpecParRegistryESProducer::produce(const DDSpecParRegistryRcd& iRecord) {
  const dd4hep::SpecParRegistry& registry = iRecord.get(m_token).specpars();
  auto product = std::make_unique<dd4hep::SpecParRegistry>();
  product->specpars.insert(registry.specpars.begin(), registry.specpars.end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDSpecParRegistryESProducer);

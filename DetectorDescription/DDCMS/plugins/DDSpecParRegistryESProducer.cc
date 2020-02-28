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
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

using namespace std;
using namespace cms;

class DDSpecParRegistryESProducer : public edm::ESProducer {
public:
  DDSpecParRegistryESProducer(const edm::ParameterSet&);
  ~DDSpecParRegistryESProducer() override;

  using ReturnType = unique_ptr<DDSpecParRegistry>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  ReturnType produce(const DDSpecParRegistryRcd&);

private:
  edm::ESGetToken<DDDetector, IdealGeometryRecord> m_token;
};

DDSpecParRegistryESProducer::DDSpecParRegistryESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this).setConsumes(m_token,
                                    edm::ESInputTag("", iConfig.getParameter<std::string>("appendToDataLabel")));
}

DDSpecParRegistryESProducer::~DDSpecParRegistryESProducer() {}

void DDSpecParRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDSpecParRegistryESProducer::ReturnType DDSpecParRegistryESProducer::produce(const DDSpecParRegistryRcd& iRecord) {
  const DDSpecParRegistry& registry = iRecord.get(m_token).specpars();
  auto product = std::make_unique<DDSpecParRegistry>();
  product->specpars.insert(registry.specpars.begin(), registry.specpars.end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDSpecParRegistryESProducer);

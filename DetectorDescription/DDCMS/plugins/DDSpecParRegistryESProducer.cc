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
#include "Geometry/Records/interface/GeometryFileRcd.h"
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
  const string m_label;
};

DDSpecParRegistryESProducer::DDSpecParRegistryESProducer(const edm::ParameterSet& iConfig)
    : m_label(iConfig.getParameter<std::string>("appendToDataLabel")) {
  setWhatProduced(this);
}

DDSpecParRegistryESProducer::~DDSpecParRegistryESProducer() {}

void DDSpecParRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDSpecParRegistryESProducer::ReturnType DDSpecParRegistryESProducer::produce(const DDSpecParRegistryRcd& iRecord) {
  edm::ESHandle<DDDetector> det;
  iRecord.getRecord<GeometryFileRcd>().get(m_label, det);

  const DDSpecParRegistry* registry = det->description()->extension<DDSpecParRegistry>();
  auto product = std::make_unique<DDSpecParRegistry>();
  product->specpars.insert(registry->specpars.begin(), registry->specpars.end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDSpecParRegistryESProducer);

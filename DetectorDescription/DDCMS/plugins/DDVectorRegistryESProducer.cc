// -*- C++ -*-
//
// Package:    DetectorDescription/DDCMS
// Class:      DDVectorRegistryESProducer
// 
/**\class DDVectorRegistryESProducer

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

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DD4hep/Detector.h"

using namespace std;
using namespace cms;

class DDVectorRegistryESProducer : public edm::ESProducer {
public:

  DDVectorRegistryESProducer(const edm::ParameterSet&);
  ~DDVectorRegistryESProducer() override;
  
  using ReturnType = unique_ptr<DDVectorRegistry>;

  static void fillDescriptions(edm::ConfigurationDescriptions&);
  
  ReturnType produce(const DDVectorRegistryRcd&);
};

DDVectorRegistryESProducer::DDVectorRegistryESProducer(const edm::ParameterSet&)
{
  setWhatProduced(this);
}

DDVectorRegistryESProducer::~DDVectorRegistryESProducer()
{
}

void
DDVectorRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDVectorRegistryESProducer::ReturnType
DDVectorRegistryESProducer::produce(const DDVectorRegistryRcd& iRecord)
{  
  edm::ESHandle<DDDetector> det;
  iRecord.getRecord<DetectorDescriptionRcd>().get(det);

  DDVectorsMap* registry = det->description->extension<DDVectorsMap>();

  auto product = std::make_unique<DDVectorRegistry>();
  product->vectors.insert(registry->begin(), registry->end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDVectorRegistryESProducer);

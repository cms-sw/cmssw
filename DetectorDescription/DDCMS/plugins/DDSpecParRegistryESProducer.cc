// -*- C++ -*-
//
// Package:    DetectorDescription/DDCMS
// Class:      DDSpecParRegistryESProducer
// 
/**\class DDSpecParRegistryESProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
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
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
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
};

DDSpecParRegistryESProducer::DDSpecParRegistryESProducer(const edm::ParameterSet&)
{
  setWhatProduced(this);
}

DDSpecParRegistryESProducer::~DDSpecParRegistryESProducer()
{
}

void
DDSpecParRegistryESProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  descriptions.addDefault(desc);
}

DDSpecParRegistryESProducer::ReturnType
DDSpecParRegistryESProducer::produce(const DDSpecParRegistryRcd& iRecord)
{  
  edm::ESHandle<DDDetector> det;
  iRecord.getRecord<DetectorDescriptionRcd>().get(det);

  DDSpecParRegistry* registry = det->description->extension<DDSpecParRegistry>();
  cout << "Got a specpar registry " << registry->specpars.size() << "\n";
  auto product = std::make_unique<DDSpecParRegistry>();
  product->specpars.insert(registry->specpars.begin(), registry->specpars.end());
  return product;
}

DEFINE_FWK_EVENTSETUP_MODULE(DDSpecParRegistryESProducer);

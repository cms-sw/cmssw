// -*- C++ -*-
//
// Package:     RecoLocalTracker/SiPixelRecHits
// Class  :     SiPixelTemplateStoreESProducer
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Tue, 08 Aug 2023 14:21:50 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CondFormats/SiPixelTransient/interface/SiPixelTemplate.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"

#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"

class SiPixelTemplateStoreESProducer : public edm::ESProducer {
public:
  SiPixelTemplateStoreESProducer(edm::ParameterSet const&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  std::unique_ptr<std::vector<SiPixelTemplateStore>> produce(const SiPixelTemplateDBObjectESProducerRcd&);

private:
  edm::ESGetToken<SiPixelTemplateDBObject, SiPixelTemplateDBObjectESProducerRcd> token_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiPixelTemplateStoreESProducer::SiPixelTemplateStoreESProducer(edm::ParameterSet const& iPSet) {
  token_ = setWhatProduced(this).consumes();
}

//
// member functions
//
std::unique_ptr<std::vector<SiPixelTemplateStore>> SiPixelTemplateStoreESProducer::produce(
    const SiPixelTemplateDBObjectESProducerRcd& iRecord) {
  auto returnValue = std::make_unique<std::vector<SiPixelTemplateStore>>();

  if (not SiPixelTemplate::pushfile(iRecord.get(token_), *returnValue)) {
    throw cms::Exception("SiPixelTemplateDBObjectFailure")
        << "Templates not filled correctly. Check the DB. Using SiPixelTemplateDBObject version "
        << iRecord.get(token_).version();
  }

  return returnValue;
}

//
// const member functions
//

//
// static member functions
//
void SiPixelTemplateStoreESProducer::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription iPSet;
  iDesc.addDefault(iPSet);
  iDesc.add("SiPixelTemplateStoreESProducer", iPSet);
}

DEFINE_FWK_EVENTSETUP_MODULE(SiPixelTemplateStoreESProducer);
;

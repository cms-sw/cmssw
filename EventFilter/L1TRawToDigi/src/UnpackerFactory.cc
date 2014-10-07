#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

EDM_REGISTER_PLUGINFACTORY(l1t::UnpackerFactoryFacility,"UnpackerFactory");

namespace l1t {
   const UnpackerFactory UnpackerFactory::instance_;

   const UnpackerFactory*
   UnpackerFactory::get()
   {
      return &instance_;
   }

   UnpackerFactory::UnpackerFactory() {};
   UnpackerFactory::~UnpackerFactory() {};

   std::auto_ptr<BaseUnpackerFactory>
   UnpackerFactory::makeUnpackerFactory(const std::string& type, const edm::ParameterSet& cfg, edm::one::EDProducerBase& prod) const
   {
      auto factory = std::auto_ptr<BaseUnpackerFactory>(UnpackerFactoryFacility::get()->create(type, cfg, prod));

      if (factory.get() == 0) {
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule")
            << "cannot find unpacker factory " << type;
      }

      return factory;
   }
}

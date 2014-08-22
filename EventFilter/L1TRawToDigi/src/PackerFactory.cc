#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

EDM_REGISTER_PLUGINFACTORY(l1t::PackerFactoryFacility,"PackerFactory");

namespace l1t {
   const PackerFactory PackerFactory::instance_;

   const PackerFactory*
   PackerFactory::get()
   {
      return &instance_;
   }

   PackerFactory::PackerFactory() {};
   PackerFactory::~PackerFactory() {};

   std::auto_ptr<BasePackerFactory>
   PackerFactory::makePackerFactory(const std::string& name, const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) const
   {
      auto factory = std::auto_ptr<BasePackerFactory>(PackerFactoryFacility::get()->create(name, cfg, cc));

      if (factory.get() == 0) {
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule")
            << "Packer factory can't find plugin " << name;
      }

      return factory;
   }
}

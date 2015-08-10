#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

EDM_REGISTER_PLUGINFACTORY(l1t::PackingSetupFactoryT,"PackingSetupFactory");

namespace l1t {
   const PackingSetupFactory PackingSetupFactory::instance_;

   std::auto_ptr<PackingSetup>
   PackingSetupFactory::make(const std::string& type) const
   {
      auto helper = std::auto_ptr<PackingSetup>(PackingSetupFactoryT::get()->create("l1t::" + type));

      if (helper.get() == 0)
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule") << "cannot find packing setup " << type;

      return helper;
   }

   void
   PackingSetupFactory::fillDescription(edm::ParameterSetDescription& desc) const
   {
      for (const auto& info: PackingSetupFactoryT::get()->available()) {
         PackingSetupFactoryT::get()->create(info.name_)->fillDescription(desc);
      }
   }
}

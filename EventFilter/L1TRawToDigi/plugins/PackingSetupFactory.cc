#include "PackingSetupFactory.h"

#include "FWCore/Utilities/interface/EDMException.h"

EDM_REGISTER_PLUGINFACTORY(l1t::PackingSetupFactoryT,"PackingSetupFactory");

namespace l1t {
   const PackingSetupFactory PackingSetupFactory::instance_;

   std::unique_ptr<PackingSetup>
   PackingSetupFactory::make(const std::string& type) const
   {
      auto helper = std::unique_ptr<PackingSetup>(PackingSetupFactoryT::get()->create("l1t::" + type));

      if (helper.get() == nullptr)
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

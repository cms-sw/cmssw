#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerSetup.h"

EDM_REGISTER_PLUGINFACTORY(l1t::UnpackerSetupFactoryT,"UnpackerSetupFactory");

namespace l1t {
   const UnpackerSetupFactory UnpackerSetupFactory::instance_;

   std::auto_ptr<UnpackerSetup>
   UnpackerSetupFactory::make(const std::string& type, edm::one::EDProducerBase& p) const
   {
      auto helper = std::auto_ptr<UnpackerSetup>(UnpackerSetupFactoryT::get()->create(type, p));

      if (helper.get() == 0)
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule") << "cannot find unpacker provider " << type;

      return helper;
   }
}

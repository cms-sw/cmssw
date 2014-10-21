#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerProvider.h"

EDM_REGISTER_PLUGINFACTORY(l1t::UnpackerProviderFactoryT,"UnpackerProviderFactory");

namespace l1t {
   const UnpackerProviderFactory UnpackerProviderFactory::instance_;

   std::auto_ptr<UnpackerProvider>
   UnpackerProviderFactory::make(const std::string& type, edm::one::EDProducerBase& p) const
   {
      auto helper = std::auto_ptr<UnpackerProvider>(UnpackerProviderFactoryT::get()->create(type, p));

      if (helper.get() == 0)
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule") << "cannot find unpacker provider " << type;

      return helper;
   }
}

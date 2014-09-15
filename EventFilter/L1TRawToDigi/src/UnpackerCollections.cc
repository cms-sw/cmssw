#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

EDM_REGISTER_PLUGINFACTORY(l1t::UnpackerCollectionsFactoryT,"UnpackerCollectionsFactory");
EDM_REGISTER_PLUGINFACTORY(l1t::UnpackerCollectionsProducesFactoryT,"UnpackerCollectionsProducesFactory");

namespace l1t {
   const UnpackerCollectionsFactory UnpackerCollectionsFactory::instance_;
   const UnpackerCollectionsProducesFactory UnpackerCollectionsProducesFactory::instance_;

   std::auto_ptr<UnpackerCollections>
   UnpackerCollectionsFactory::makeUnpackerCollections(const std::string& type, edm::Event& e) const
   {
      auto collection = std::auto_ptr<UnpackerCollections>(UnpackerCollectionsFactoryT::get()->create(type, e));

      if (collection.get() == 0)
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule") << "cannot find unpacker collection " << type;

      return collection;
   }

   std::auto_ptr<UnpackerCollectionsProduces>
   UnpackerCollectionsProducesFactory::makeUnpackerCollectionsProduces(const std::string& type, edm::one::EDProducerBase& p) const
   {
      auto helper = std::auto_ptr<UnpackerCollectionsProduces>(UnpackerCollectionsProducesFactoryT::get()->create(type, p));

      if (helper.get() == 0)
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule") << "cannot find unpacker collection helper " << type;

      return helper;
   }
}

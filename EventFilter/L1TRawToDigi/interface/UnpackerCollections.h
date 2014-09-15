#ifndef UnpackerCollections_h
#define UnpackerCollections_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {
   class Event;
   namespace one {
      class EDProducerBase;
   }
}

namespace l1t {
   class UnpackerCollections {
      public:
         UnpackerCollections(edm::Event& e) : event_(e) {};
         virtual ~UnpackerCollections() {};
      protected:
         edm::Event& event_;
   };

   class UnpackerCollectionsProduces {
      public:
         UnpackerCollectionsProduces(edm::one::EDProducerBase&) {};
   };

   typedef UnpackerCollections*(coll_fct)(edm::Event&);
   typedef edmplugin::PluginFactory<coll_fct> UnpackerCollectionsFactoryT;

   typedef UnpackerCollectionsProduces*(prod_fct)(edm::one::EDProducerBase&);
   typedef edmplugin::PluginFactory<prod_fct> UnpackerCollectionsProducesFactoryT;

   class UnpackerCollectionsFactory {
      public:
         static const UnpackerCollectionsFactory* get() { return &instance_; };
         std::auto_ptr<UnpackerCollections> makeUnpackerCollections(const std::string&, edm::Event&) const;
      private:
         UnpackerCollectionsFactory() {};
         static const UnpackerCollectionsFactory instance_;
   };

   class UnpackerCollectionsProducesFactory {
      public:
         static const UnpackerCollectionsProducesFactory* get() { return &instance_; };
         std::auto_ptr<UnpackerCollectionsProduces> makeUnpackerCollectionsProduces(const std::string&, edm::one::EDProducerBase&) const;
      private:
         UnpackerCollectionsProducesFactory() {};
         static const UnpackerCollectionsProducesFactory instance_;
   };
}

#define DEFINE_L1TUNPACKER_COLLECTION(type) \
   DEFINE_EDM_PLUGIN(l1t::UnpackerCollectionsFactoryT,type,#type)

#define DEFINE_L1TUNPACKER_COLLECTION_PRODUCES(type) \
   DEFINE_EDM_PLUGIN(l1t::UnpackerCollectionsProducesFactoryT,type,#type)

#endif

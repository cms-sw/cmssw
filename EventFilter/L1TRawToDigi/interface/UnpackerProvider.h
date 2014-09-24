#ifndef UnpackerProvider_h
#define UnpackerProvider_h

#include <map>
#include <tuple>

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace edm {
   class Event;
   namespace one {
      class EDProducerBase;
   }
}

namespace l1t {
   typedef std::tuple<int, int, int, int> UnpackerVersion;
   typedef std::map<UnpackerVersion, std::shared_ptr<Unpacker>> UnpackerMap;

   class UnpackerProvider {
      public:
         UnpackerProvider(edm::one::EDProducerBase&) {};

         virtual UnpackerMap getUnpackers() = 0;
         virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event&) = 0;
   };

   typedef UnpackerProvider*(prov_fct)(edm::one::EDProducerBase&);
   typedef edmplugin::PluginFactory<prov_fct> UnpackerProviderFactoryT;

   class UnpackerProviderFactory {
      public:
         static const UnpackerProviderFactory* get() { return &instance_; };
         std::auto_ptr<UnpackerProvider> make(const std::string&, edm::one::EDProducerBase&) const;
      private:
         UnpackerProviderFactory() {};
         static const UnpackerProviderFactory instance_;
   };
}

#define DEFINE_L1T_UNPACKER_PROVIDER(type) \
   DEFINE_EDM_PLUGIN(l1t::UnpackerProviderFactoryT,type,#type)

#endif

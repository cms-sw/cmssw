#ifndef UnpackerSetup_h
#define UnpackerSetup_h

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
   // Mapping of block id to unpacker.  Different for each set of (FED, AMC, Firmware) ids.
   typedef std::map<int, std::shared_ptr<Unpacker>> UnpackerMap;

   class UnpackerSetup {
      public:
         UnpackerSetup(edm::one::EDProducerBase&) {};

         virtual UnpackerMap getUnpackers(int, int, int) = 0;
         virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event&) = 0;
   };

   typedef UnpackerSetup*(prov_fct)(edm::one::EDProducerBase&);
   typedef edmplugin::PluginFactory<prov_fct> UnpackerSetupFactoryT;

   class UnpackerSetupFactory {
      public:
         static const UnpackerSetupFactory* get() { return &instance_; };
         std::auto_ptr<UnpackerSetup> make(const std::string&, edm::one::EDProducerBase&) const;
      private:
         UnpackerSetupFactory() {};
         static const UnpackerSetupFactory instance_;
   };
}

#define DEFINE_L1T_UNPACKER_PROVIDER(type) \
   DEFINE_EDM_PLUGIN(l1t::UnpackerSetupFactoryT,type,#type)

#endif

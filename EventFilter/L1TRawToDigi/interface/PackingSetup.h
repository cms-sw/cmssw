#ifndef PackingSetup_h
#define PackingSetup_h

#include <map>

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace edm {
   class ConsumesCollector;
   class Event;
   class ParameterSet;
   namespace one {
      class EDProducerBase;
   }
}

namespace l1t {
   // Mapping of AMC id to list of unpackers.  Different for each set of (FED, Firmware) ids.
   typedef std::map<int, Packers> PackerMap;
   // Mapping of block id to unpacker.  Different for each set of (FED, AMC, Firmware) ids.
   typedef std::map<int, std::shared_ptr<Unpacker>> UnpackerMap;

   class PackingSetup {
      public:
         PackingSetup() {};
         virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet&, edm::ConsumesCollector&) = 0;
         virtual void registerProducts(edm::one::EDProducerBase&) = 0;

         // Get a map of AMC ↔ list of packing functions for a specific FED, FW combination
         virtual PackerMap getPackers(int, int) = 0;

         // Get a map of Block IDs ↔ unpacker for a specific FED, AMC, FW combination
         virtual UnpackerMap getUnpackers(int, int, int) = 0;
         virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event&) = 0;
   };

   typedef PackingSetup*(prov_fct)();
   typedef edmplugin::PluginFactory<prov_fct> PackingSetupFactoryT;

   class PackingSetupFactory {
      public:
         static const PackingSetupFactory* get() { return &instance_; };
         std::auto_ptr<PackingSetup> make(const std::string&) const;
      private:
         PackingSetupFactory() {};
         static const PackingSetupFactory instance_;
   };
}

#define DEFINE_L1T_PACKING_SETUP(type) \
   DEFINE_EDM_PLUGIN(l1t::PackingSetupFactoryT,type,#type)

#endif

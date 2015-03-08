#ifndef UnpackerFactory_h
#define UnpackerFactory_h

#include <memory>
#include <unordered_map>
#include <vector>

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "EventFilter/L1TRawToDigi/interface/BaseUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"

namespace edm {
   class Event;
   class ParameterSet;
   namespace one {
      class EDProducerBase;
   }
}

namespace l1t {
   typedef std::pair<BlockId, std::shared_ptr<l1t::BaseUnpacker>> UnpackerItem;
   typedef std::unordered_map<BlockId, std::shared_ptr<l1t::BaseUnpacker>> UnpackerMap;

   inline uint32_t pop(const unsigned char* ptr, unsigned& idx) {
     uint32_t res = ptr[idx + 0] | (ptr[idx + 1] << 8) | (ptr[idx + 2] << 16) | (ptr[idx + 3] << 24);
     idx += 4;
     return res;
   };

   class BaseUnpackerFactory {
      public:
         virtual std::vector<UnpackerItem> create(edm::Event&, const unsigned&, const int fedid) = 0;
   };

   typedef BaseUnpackerFactory*(fct)(const edm::ParameterSet&, edm::one::EDProducerBase&);
   typedef edmplugin::PluginFactory<fct> UnpackerFactoryFacility;

   class UnpackerFactory {
      public:
         ~UnpackerFactory();

         static const UnpackerFactory* get();

         std::auto_ptr<BaseUnpackerFactory> makeUnpackerFactory(const std::string&, const edm::ParameterSet&, edm::one::EDProducerBase&) const;

      private:
         UnpackerFactory();
         static const UnpackerFactory instance_;
   };
}

#define DEFINE_L1TUNPACKER(type) \
   DEFINE_EDM_PLUGIN(l1t::UnpackerFactoryFacility,type,#type)

#endif

#ifndef PackerFactory_h
#define PackerFactory_h

#include <memory>
#include <vector>

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "EventFilter/L1TRawToDigi/interface/BasePacker.h"

namespace edm {
   class ConsumesCollector;
   class ParameterSet;
   namespace one {
      class EDProducerBase;
   }
}

namespace l1t {
   typedef std::vector<std::shared_ptr<l1t::BasePacker>> PackerList;

   class BasePackerFactory {
      public:
         virtual PackerList create(const unsigned&, const int fedid) = 0;
   };

   typedef BasePackerFactory*(fun)(const edm::ParameterSet&, edm::ConsumesCollector&);
   typedef edmplugin::PluginFactory<fun> PackerFactoryFacility;

   class PackerFactory {
      public:
         ~PackerFactory();

         static const PackerFactory* get();

         std::auto_ptr<BasePackerFactory> makePackerFactory(const edm::ParameterSet&, edm::ConsumesCollector&) const;

      private:
         PackerFactory();
         static const PackerFactory instance_;
   };
}

#define DEFINE_L1TPACKER(type) \
   DEFINE_EDM_PLUGIN(l1t::PackerFactoryFacility,type,#type)

#endif

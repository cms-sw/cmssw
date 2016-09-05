#ifndef EventFilter_L1TRawToDigi_PackingSetupFactory_h
#define EventFilter_L1TRawToDigi_PackingSetupFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

namespace l1t {
   typedef PackingSetup*(prov_fct)();
   typedef edmplugin::PluginFactory<prov_fct> PackingSetupFactoryT;

   class PackingSetupFactory {
      public:
         static const PackingSetupFactory* get() { return &instance_; };
         std::auto_ptr<PackingSetup> make(const std::string&) const;
         void fillDescription(edm::ParameterSetDescription&) const;
      private:
         PackingSetupFactory() {};
         static const PackingSetupFactory instance_;
   };
}

#define DEFINE_L1T_PACKING_SETUP(type) \
   DEFINE_EDM_PLUGIN(l1t::PackingSetupFactoryT,type,#type)

#endif

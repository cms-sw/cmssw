#ifndef EventFilter_L1TRawToDigi_PackerFactory_h
#define EventFilter_L1TRawToDigi_PackerFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   typedef Packer*(pack_fct)();
   typedef edmplugin::PluginFactory<pack_fct> PackerFactoryT;

   class PackerFactory {
      public:
         inline static const PackerFactory* get() { return &instance_; };
         std::shared_ptr<Packer> make(const std::string&) const;
      private:
         PackerFactory() {};
         static const PackerFactory instance_;
   };
}

#define DEFINE_L1T_PACKER(type) \
   DEFINE_EDM_PLUGIN(l1t::PackerFactoryT,type,#type)

#endif

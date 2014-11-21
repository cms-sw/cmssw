#ifndef Packer_h
#define Packer_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace edm {
   class Event;
}

namespace l1t {
   class L1TDigiToRaw;

   class Packer {
      public:
         virtual Blocks pack(const edm::Event&, const PackerTokens*) = 0;
   };

   typedef std::vector<std::shared_ptr<Packer>> Packers;

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

#ifndef Unpacker_h
#define Unpacker_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "EventFilter/L1TRawToDigi/interface/Block.h"

namespace l1t {
   class UnpackerCollections;

   void getBXRange(int nbx, int& first, int& last);

   class Unpacker {
      public:
         virtual bool unpack(const Block& block, UnpackerCollections *coll) = 0;
   };

   typedef Unpacker*(unpack_fct)();
   typedef edmplugin::PluginFactory<unpack_fct> UnpackerFactoryT;

   class UnpackerFactory {
      public:
         inline static const UnpackerFactory* get() { return &instance_; };
         std::shared_ptr<Unpacker> make(const std::string&) const;
      private:
         UnpackerFactory() {};
         static const UnpackerFactory instance_;
   };
}

#define DEFINE_L1T_UNPACKER(type) \
   DEFINE_EDM_PLUGIN(l1t::UnpackerFactoryT,type,#type)

#endif

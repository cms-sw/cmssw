#ifndef Unpacker_h
#define Unpacker_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace l1t {
   inline uint32_t pop(const unsigned char* ptr, unsigned& idx) {
     uint32_t res = ptr[idx + 0] | (ptr[idx + 1] << 8) | (ptr[idx + 2] << 16) | (ptr[idx + 3] << 24);
     idx += 4;
     return res;
   };

   class UnpackerCollections;

   class Unpacker {
      public:
         virtual bool unpack(
               const unsigned block_id,
               const unsigned size,
               const unsigned char *data,
               UnpackerCollections *coll) = 0;
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

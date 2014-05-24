#ifndef UnpackerFactory_h
#define UnpackerFactory_h

#include <memory>
#include <unordered_map>
#include <vector>

#include "EventFilter/L1TRawToDigi/interface/BaseUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"

namespace l1t {
   typedef std::pair<BlockId, std::shared_ptr<l1t::BaseUnpacker>> UnpackerItem;
   typedef std::unordered_map<BlockId, std::shared_ptr<l1t::BaseUnpacker>> UnpackerMap;

   inline uint32_t pop(const unsigned char* ptr, unsigned& idx) {
     uint32_t res = ptr[idx + 0] | (ptr[idx + 1] << 8) | (ptr[idx + 2] << 16) | (ptr[idx + 3] << 24);
     idx += 4;
     return res;
   };

   class UnpackerFactory {
      public:
         static UnpackerMap createUnpackers(unsigned fw, const int fedid);

      private:
         virtual std::vector<UnpackerItem> create(unsigned fw, const int fedid) = 0;

         static std::vector<UnpackerFactory*> createFactories();
         static std::vector<UnpackerFactory*> factories_;
   };
}

#endif

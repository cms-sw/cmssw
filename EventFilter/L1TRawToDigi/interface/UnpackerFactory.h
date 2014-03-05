#ifndef UnpackerFactory_h
#define UnpackerFactory_h

#include <unordered_map>
#include <vector>

#include "CondFormats/L1TObjects/interface/FirmwareVersion.h"
#include "EventFilter/L1TRawToDigi/interface/BaseUnpacker.h"

namespace l1t {
   typedef uint32_t BlockId;
   typedef std::unordered_map<BlockId, l1t::BaseUnpacker*> Unpackers;

   class UnpackerFactory {
      public:
         static Unpackers createUnpackers(const FirmwareVersion&);

      private:
         virtual bool hasUnpackerFor(const FirmwareVersion&) = 0;
         virtual std::pair<BlockId, BaseUnpacker*> create(const FirmwareVersion&) = 0;

         static std::vector<UnpackerFactory*> createFactories();
         static std::vector<UnpackerFactory*> factories_;
   };
}

#endif

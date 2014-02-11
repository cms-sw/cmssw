#include "EventFilter/L1TRawToDigi/interface/BaseUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   std::vector<UnpackerFactory*> UnpackerFactory::factories_;

   std::unordered_map<BlockId, BaseUnpacker*>
   UnpackerFactory::createUnpackers(const FirmwareVersion &fw)
   {
      std::unordered_map<BlockId, BaseUnpacker*> res;
      for (const auto& f: factories_) {
         if (f->hasUnpackerFor(fw))
            res.insert(f->create(fw));
      }
      return res;
   }

   void
   UnpackerFactory::registerUnpackerFactory(UnpackerFactory *f)
   {
      factories_.push_back(f);
   }
}

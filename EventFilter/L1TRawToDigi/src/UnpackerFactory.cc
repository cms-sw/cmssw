#include "EventFilter/L1TRawToDigi/interface/BaseUnpacker.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "implementations/JetUnpacker.h"

namespace l1t {
   std::vector<UnpackerFactory*> UnpackerFactory::factories_ = UnpackerFactory::createFactories();

   std::vector<UnpackerFactory*> UnpackerFactory::createFactories()
   {
      std::vector<UnpackerFactory*> res;
      res.push_back(new JetUnpackerFactory());
      return res;
   }

   std::unordered_map<BlockId, BaseUnpacker*>
   UnpackerFactory::createUnpackers(const FirmwareVersion &fw)
   {
      std::cout << factories_.size() << std::endl;
      std::unordered_map<BlockId, BaseUnpacker*> res;
      for (const auto& f: factories_) {
         if (f->hasUnpackerFor(fw))
            res.insert(f->create(fw));
      }
      return res;
   }
}

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

   UnpackerMap
   UnpackerFactory::createUnpackers(const FirmwareVersion &fw, const int fedid)
   {
      UnpackerMap res;
      for (const auto& f: factories_) {
         for (auto& i: f->create(fw, fedid)) {
            if (res.find(i.first) == res.end()) {
               res.insert(i);
            } else {
               // FIXME make proper edm error message
               throw;
            }
         }
      }
      return res;
   }
}

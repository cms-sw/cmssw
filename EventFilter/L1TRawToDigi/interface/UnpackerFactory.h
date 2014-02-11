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
         static void registerUnpackerFactory(UnpackerFactory *f);

      private:
         virtual bool hasUnpackerFor(const FirmwareVersion&) = 0;
         virtual std::pair<BlockId, BaseUnpacker*> create(const FirmwareVersion&) = 0;

         static std::vector<UnpackerFactory*> factories_;
   };

   template<typename T>
   class UnpackerFactoryRegistration {
      private:
         UnpackerFactoryRegistration() { reg; };
         static bool reg;
         static bool init() {
            UnpackerFactory::registerUnpackerFactory(new T);
            return true;
         };

         static UnpackerFactoryRegistration<T> singleton;
   };

   template<typename T> bool UnpackerFactoryRegistration<T>::reg = UnpackerFactoryRegistration<T>::init();
   template<typename T> UnpackerFactoryRegistration<T> UnpackerFactoryRegistration<T>::singleton;
}

#endif

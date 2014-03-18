#include "DataFormats/L1Trigger/interface/Jet.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "JetUnpacker.h"

namespace l1t {
   class JetUnpacker : public BaseUnpacker {
      public:
         virtual bool unpack(const unsigned char *data, const unsigned size) {
            return true;
         };

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getJetCollection();
         };
      private:
         JetBxCollection* res;
   };

   bool JetUnpackerFactory::hasUnpackerFor(const FirmwareVersion& fw, const int fedid) {
      return true;
   };

   std::pair<BlockId, BaseUnpacker*> JetUnpackerFactory::create(const FirmwareVersion& fw, const int fedid) {
      return std::pair<BlockId, BaseUnpacker*>(0xF, new JetUnpacker());
   };
}

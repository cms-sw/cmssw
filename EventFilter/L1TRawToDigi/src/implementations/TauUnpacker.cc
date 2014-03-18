#include "DataFormats/L1Trigger/interface/Tau.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class TauUnpacker : public BaseUnpacker {
      public:
         virtual bool unpack(const unsigned char *data, const unsigned size) {
            return true;
         };

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getTauCollection();
         };
      private:
         TauBxCollection* res;
   };

   class TauUnpackerFactory : UnpackerFactory {
      public:
         virtual bool hasUnpackerFor(const FirmwareVersion& fw, const int fedid) {
            return true;
         };

         virtual std::pair<BlockId, BaseUnpacker*> create(const FirmwareVersion& fw, const int fedid) {
            return std::pair<BlockId, BaseUnpacker*>(0x1, new TauUnpacker());
         };
   };
}

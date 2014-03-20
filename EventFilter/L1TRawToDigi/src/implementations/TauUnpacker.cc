#include "DataFormats/L1Trigger/interface/Tau.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class TauUnpacker : public BaseUnpacker {
      public:
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {
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
         virtual std::vector<UnpackerItem> create(const FirmwareVersion& fw, const int fedid) {
            return {std::make_pair(0x1, std::shared_ptr<BaseUnpacker>(new TauUnpacker()))};
         };
   };
}

#include "DataFormats/L1Trigger/interface/Jet.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

#include "JetUnpacker.h"

namespace l1t {
   class JetUnpacker : public BaseUnpacker {
      public:
         virtual bool unpack(const unsigned char *data, const unsigned block_id, const unsigned size) {
            return true;
         };

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getJetCollection();
         };
      private:
         JetBxCollection* res;
   };

   std::vector<UnpackerItem> JetUnpackerFactory::create(const FirmwareVersion& fw, const int fedid) {
      return {std::make_pair(0xF, std::shared_ptr<BaseUnpacker>(new JetUnpacker()))};
   };
}

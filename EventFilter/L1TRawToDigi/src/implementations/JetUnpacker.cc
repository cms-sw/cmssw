#include "DataFormats/L1Trigger/interface/Jet.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class JetUnpacker : public BaseUnpacker {
      public:
         virtual bool unpack(const unsigned char *data, const unsigned size) {
            std::cout << "I see a jet with size " << size << std::endl;
            return true;
         };

         virtual void setCollections(UnpackerCollections& coll) {
            res = coll.getJetCollection();
         };
      private:
         JetBxCollection* res;
   };

   class JetUnpackerFactory : UnpackerFactory, UnpackerFactoryRegistration<JetUnpackerFactory> {
      private:
         virtual bool hasUnpackerFor(const FirmwareVersion& fw) {
            return true;
         };

         virtual std::pair<BlockId, BaseUnpacker*> create(const FirmwareVersion& fw) {
            return std::pair<BlockId, BaseUnpacker*>(0x1, new JetUnpacker());
         };
   };
}

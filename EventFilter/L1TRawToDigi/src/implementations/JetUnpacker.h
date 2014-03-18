#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class JetUnpackerFactory : public UnpackerFactory {
      public:
         virtual bool hasUnpackerFor(const FirmwareVersion& fw, const int fedid);
         virtual std::pair<BlockId, BaseUnpacker*> create(const FirmwareVersion& fw, const int fedid);
   };
}

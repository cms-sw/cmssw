#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class EtSumUnpackerFactory : public UnpackerFactory {
      public:
         virtual std::vector<UnpackerItem> create(unsigned fw, const int fedid);
   };
}

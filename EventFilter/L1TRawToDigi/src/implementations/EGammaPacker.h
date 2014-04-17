#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class EGammaPackerFactory : public PackerFactory {
      public:
         virtual PackerList create(const edm::ParameterSet& cfg, const FirmwareVersion& fw, const int fedid);
   };
}

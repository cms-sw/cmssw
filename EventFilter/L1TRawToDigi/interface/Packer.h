#ifndef EventFilter_L1TRawToDigi_Packer_h
#define EventFilter_L1TRawToDigi_Packer_h

#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace edm {
   class Event;
}

namespace l1t {
   class L1TDigiToRaw;

   class Packer {
      public:
         virtual Blocks pack(const edm::Event&, const PackerTokens*) = 0;
   };

   typedef std::vector<std::shared_ptr<Packer>> Packers;
}

#endif

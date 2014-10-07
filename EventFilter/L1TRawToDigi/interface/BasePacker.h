#ifndef BasePacker_h
#define BasePacker_h

#include "EventFilter/L1TRawToDigi/interface/Block.h"

namespace edm {
   class Event;
   class ParameterSet;
}

namespace l1t {
   class L1TDigiToRaw;

   class BasePacker {
      public:
         virtual Blocks pack(const edm::Event&) = 0;
   };
}

#endif

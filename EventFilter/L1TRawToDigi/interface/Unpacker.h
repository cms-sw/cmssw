#ifndef EventFilter_L1TRawToDigi_Unpacker_h
#define EventFilter_L1TRawToDigi_Unpacker_h

#include "EventFilter/L1TRawToDigi/interface/Block.h"

namespace l1t {
   class UnpackerCollections;

   void getBXRange(int nbx, int& first, int& last);

   class Unpacker {
      public:
         virtual bool unpack(const Block& block, UnpackerCollections *coll) = 0;
   };
}

#endif

#ifndef BaseUnpacker_h
#define BaseUnpacker_h

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   class BaseUnpacker {
      public:
         // Returns successful read
         virtual bool unpack(const unsigned char *data, const unsigned blockid, const unsigned size) = 0;
         // Obtain the collection(s) to unpack into
         virtual void setCollections(UnpackerCollections& coll) = 0;
   };
}

#endif

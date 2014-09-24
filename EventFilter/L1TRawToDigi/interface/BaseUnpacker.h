#ifndef BaseUnpacker_h
#define BaseUnpacker_h

#include "UnpackerCollections.h"

namespace l1t {
   class BaseUnpacker {
      public:
         virtual bool unpack(
               const unsigned block_id,
               const unsigned size,
               const unsigned char *data,
               UnpackerCollections *coll) = 0;
   };
}

#endif

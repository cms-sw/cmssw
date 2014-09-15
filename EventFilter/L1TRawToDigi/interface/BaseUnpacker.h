#ifndef BaseUnpacker_h
#define BaseUnpacker_h

#include "UnpackerCollections.h"

namespace l1t {
   class BaseUnpacker {
      public:
         BaseUnpacker(UnpackerCollections* coll) : collections_(coll) {};
         // Returns successful read
         virtual bool unpack(const unsigned char *data, const unsigned blockid, const unsigned size) = 0;
      protected:
         UnpackerCollections* collections_;
   };
}

#endif

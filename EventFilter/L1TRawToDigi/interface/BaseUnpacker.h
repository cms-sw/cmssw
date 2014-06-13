#ifndef BaseUnpacker_h
#define BaseUnpacker_h

namespace l1t {
   class BaseUnpacker {
      public:
         // Returns successful read
         virtual bool unpack(const unsigned char *data, const unsigned blockid, const unsigned size) = 0;
   };
}

#endif

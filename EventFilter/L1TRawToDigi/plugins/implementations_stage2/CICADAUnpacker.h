#ifndef EventFilter_L1TRawToDigi_CICADAUnpacker_h
#define EventFilter_L1TRawToDigi_CICADAUnpacker_h

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "CaloLayer1Collections.h"

namespace l1t {
  namespace stage2 {
    class CICADAUnpacker : public Unpacker {
    public:
      bool unpack(const Block& block, UnpackerCollections* coll) override;

    private:
      static constexpr unsigned int caloCrateCicadaBitsPattern = 0xF0000000;  //first 4 bits of the words are CICADA
    };
  }  // namespace stage2
}  // namespace l1t

#endif

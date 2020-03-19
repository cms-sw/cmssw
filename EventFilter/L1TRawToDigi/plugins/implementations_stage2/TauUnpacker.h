#ifndef L1T_PACKER_STAGE2_TAUUNPACKER_H
#define L1T_PACKER_STAGE2_TAUUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class TauUnpacker : public Unpacker {
    public:
      TauUnpacker();
      ~TauUnpacker() override{};

      bool unpack(const Block& block, UnpackerCollections* coll) override;

      inline void setTauCopy(const unsigned int copy) { TauCopy_ = copy; };

    private:
      unsigned int TauCopy_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

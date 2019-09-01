#ifndef L1T_PACKER_STAGE2_JETUNPACKER_H
#define L1T_PACKER_STAGE2_JETUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class JetUnpacker : public Unpacker {
    public:
      JetUnpacker();
      ~JetUnpacker() override{};

      bool unpack(const Block& block, UnpackerCollections* coll) override;

      inline void setJetCopy(const unsigned int copy) { JetCopy_ = copy; };

    private:
      unsigned int JetCopy_;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

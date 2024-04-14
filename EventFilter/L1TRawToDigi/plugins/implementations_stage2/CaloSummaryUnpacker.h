#ifndef L1T_PACKER_STAGE2_CaloSummaryUnpacker_H
#define L1T_PACKER_STAGE2_CaloSummaryUnpacker_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    class CaloSummaryUnpacker : public Unpacker {
    public:
      CaloSummaryUnpacker() = default;
      ~CaloSummaryUnpacker() override = default;

      bool unpack(const Block& block, UnpackerCollections* coll) override;
      float processBitsToScore(const unsigned int[]);
      
      static constexpr unsigned short numCICADAWords = 4;  // We have 4 words/frames that contain CICADA bits
      static constexpr unsigned int nFramesPerEvent = 6;           //Calo Summary outputs 6 32 bit words (or frames in uGT parlance) per event.
      static constexpr unsigned int cicadaBitsPattern =
          0xF0000000;  //first 4 bits of the first 4 words/frames are CICADA
    };
  }  // namespace stage2
}  // namespace l1t
#endif

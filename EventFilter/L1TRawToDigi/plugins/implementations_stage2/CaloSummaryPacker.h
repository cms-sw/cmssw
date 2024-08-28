#ifndef PACKER_STAGE2_CALOSUMMARYPACKER_H
#define PACKER_STAGE2_CALOSUMMARYPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
  namespace stage2 {
    class CaloSummaryPacker : public Packer {
    private:
      std::vector<uint32_t> generateCICADAWordsFromScore(float);

    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;
    };
  }  // namespace stage2
}  // namespace l1t

#endif

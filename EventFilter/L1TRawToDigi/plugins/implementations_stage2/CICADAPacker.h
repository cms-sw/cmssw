#ifndef L1T_PACKER_STAGE2_CICADAPACKER_H
#define L1T_PACKER_STAGE2_CICADAPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "CaloLayer1Tokens.h"

namespace l1t {
  namespace stage2 {
    class CICADAPacker : public Packer {
    public:
      Blocks pack(const edm::Event&, const PackerTokens*) override;

    private:
      std::vector<uint32_t> makeCICADAWordsFromScore(float);
    };
  }  // namespace stage2
}  // namespace l1t

#endif

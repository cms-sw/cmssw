#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "CaloTokens.h"

#include "L1TStage2Layer2Constants.h"
#include "ZDCPacker.h"
#include "GTSetup.h"

namespace l1t {
  namespace stage2 {
    Blocks ZDCPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      edm::Handle<EtSumBxCollection> ZDCSums;
      event.getByToken(static_cast<const CommonTokens*>(toks)->getEtSumZDCToken(), ZDCSums);

      std::vector<uint32_t> load;
      int nBx = 0;

      for (int i = ZDCSums->getFirstBX(); i <= ZDCSums->getLastBX(); ++i) {
        int zdc_mask = 0x3FF;
        uint32_t empty_word = 0;
        uint32_t zdcm_word = 0;
        uint32_t zdcp_word = 0;

        for (auto j = ZDCSums->begin(i); j != ZDCSums->end(i); ++j) {
          uint32_t word = std::min(j->hwPt(), zdc_mask);

          if (j->getType() == l1t::EtSum::kZDCM)
            zdcm_word |= word;
          if (j->getType() == l1t::EtSum::kZDCP)
            zdcp_word |= word;
        }
        load.push_back(empty_word);
        load.push_back(zdcm_word);
        load.push_back(zdcp_word);

        //pad with zeros to fill out block; must do this for each BX
        while (load.size() - nBx * zdc::nOutputFramePerBX < zdc::nOutputFramePerBX)
          load.push_back(0);
        nBx++;
      }

      return {Block(b1_, load)};
    }
  }  // namespace stage2
}  // namespace l1t

// DEFINE_L1T_PACKER(l1t::stage2::CaloEtSumZDCPacker);
DEFINE_L1T_PACKER(l1t::stage2::GTEtSumZDCPacker);

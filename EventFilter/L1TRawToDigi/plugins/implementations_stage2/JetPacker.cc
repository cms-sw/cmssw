#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "CaloTokens.h"

#include "L1TStage2Layer2Constants.h"
#include "JetPacker.h"

namespace l1t {
  namespace stage2 {
    Blocks JetPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      edm::Handle<JetBxCollection> jets;
      event.getByToken(static_cast<const CommonTokens*>(toks)->getJetToken(), jets);

      std::vector<uint32_t> load1, load2;

      // loop over BX
      int nBx = 0;
      for (int i = jets->getFirstBX(); i <= jets->getLastBX(); ++i) {
        // get jets from this BX
        for (auto j = jets->begin(i); j != jets->end(i); ++j) {
          uint32_t packed_eta = abs(j->hwEta()) & 0x7F;
          if (j->hwEta() < 0) {
            packed_eta = (128 - packed_eta) | 1 << 7;
          }

          uint32_t word =
              std::min(j->hwPt(), 0x7FF) | packed_eta << 11 | (j->hwPhi() & 0xFF) << 19 | (j->hwQual() & 0x7) << 27;

          // Depending on the number, push onto appropriate link
          if (load1.size() - nBx * l1t::stage2::layer2::demux::nJetPerLink < l1t::stage2::layer2::demux::nJetPerLink)
            load1.push_back(word);
          else
            load2.push_back(word);
        }

        // push zeroes if jets are missing must do this for each BX and each link
        while (load1.size() - nBx * l1t::stage2::layer2::demux::nOutputFramePerBX <
               l1t::stage2::layer2::demux::nOutputFramePerBX)
          load1.push_back(0);
        while (load2.size() - nBx * l1t::stage2::layer2::demux::nOutputFramePerBX <
               l1t::stage2::layer2::demux::nOutputFramePerBX)
          load2.push_back(0);

        nBx++;
      }

      return {Block(b1_, load1), Block(b2_, load2)};
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKER(l1t::stage2::GTJetPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloJetPacker);

#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"
#include "GMTTokens.h"
#include "MuonPacker.h"

namespace l1t {
  namespace stage2 {
    Blocks MuonPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      edm::Handle<MuonBxCollection> muons;
      event.getByToken(static_cast<const CommonTokens*>(toks)->getMuonToken(), muons);

      PayloadMap payloadMap;

      for (int bx = muons->getFirstBX(); bx <= muons->getLastBX(); ++bx) {
          packBx(payloadMap, muons, bx);
      }

      Blocks blocks;
      // push everything in the blocks vector
      for (auto& kv : payloadMap) {
        //cout << kv.first << ":  " << kv.second.size() << kv.second[0] << "\n";
        blocks.push_back(Block(kv.first, kv.second));
      }
      return blocks;
    }

    void MuonPacker::packBx(PayloadMap& payloadMap, const edm::Handle<MuonBxCollection>& muons, int bx) {
      // the first word in every BX and every block id is 0
      for (unsigned int blkId = b1_; blkId < b1_ + 7; blkId += 2) {
        payloadMap[blkId].push_back(0);
      }

      unsigned int blkId = b1_;
      auto mu { muons->begin(bx) };
      uint32_t shared_word { 0 };
      uint32_t mu1_msw { 0 };
      uint32_t mu2_msw { 0 };
      uint32_t mu1_lsw { 0 };
      uint32_t mu2_lsw { 0 };
      // Slightly convoluted logic to account for the Run-3 muon readout record:
      // To make space for displacement information we moved the raw (i.e. non-extrapolated) eta value to the second "spare" word
      // in the block which we call "shared word". So the logic below needs to be aware if it is operating on the first or second
      // muon in the block in order tp place the eta value in the right place in the shared word. Additionally the logic needs to
      // wait for the second muon in the block before filling the payload map because the shared word goes in first.
      for (int muCtr = 1; muCtr <= 8; ++muCtr) {

        if (mu != muons->end(bx)) {
          MuonRawDigiTranslator::generatePackedDataWords(*mu, shared_word, mu2_lsw, mu2_msw, fedId_, fwId_, 2-(muCtr%2));
          ++mu;
        }

        // If we're remaining in the current block the muon we just packed is the first one in the block.
        // If not we add both muons to the payload map and go to the next block.
        // TODO: Handle case when there's only one muon in the block.
        if (muCtr % 2 == 1) {
          mu1_lsw = mu2_lsw;
          mu1_msw = mu2_msw;
        } else {
          payloadMap[blkId].push_back(shared_word);
          payloadMap[blkId].push_back(mu1_lsw);
          payloadMap[blkId].push_back(mu1_msw);
          payloadMap[blkId].push_back(mu2_lsw);
          payloadMap[blkId].push_back(mu2_msw);

          blkId += 2;

          shared_word = 0;
          mu1_lsw = 0;
          mu1_msw = 0;
        }
        mu2_lsw = 0;
        mu2_msw = 0;
      }
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKER(l1t::stage2::GTMuonPacker);
DEFINE_L1T_PACKER(l1t::stage2::GMTMuonPacker);


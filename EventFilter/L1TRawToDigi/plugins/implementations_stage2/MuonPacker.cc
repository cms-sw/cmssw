#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "GMTTokens.h"
#include "MuonPacker.h"

namespace l1t {
  namespace stage2 {
    Blocks MuonPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      GMTOutputObjectMap gmtObjMap;
      std::pair<int, int> muonBx = getMuons(gmtObjMap, event, static_cast<const CommonTokens*>(toks)->getMuonToken());
      std::pair<int, int> muonShowerBx{0, 0};
      if ((fedId_ == 1402 && fwId_ >= 0x7000000) || (fedId_ == 1404 && fwId_ >= 0x00010f01)) {
        muonShowerBx = getMuonShowers(gmtObjMap, event, static_cast<const CommonTokens*>(toks)->getMuonShowerToken());
      }

      PayloadMap payloadMap;

      packBx(gmtObjMap, muonBx.first, muonBx.second, muonShowerBx.first, muonShowerBx.second, payloadMap);

      Blocks blocks;
      // push everything in the blocks vector
      for (auto& kv : payloadMap) {
        blocks.push_back(Block(kv.first, kv.second));
      }
      return blocks;
    }

    std::pair<int, int> MuonPacker::getMuonShowers(GMTOutputObjectMap& objMap,
                                                   const edm::Event& event,
                                                   const edm::EDGetTokenT<MuonShowerBxCollection>& showerToken) {
      edm::Handle<MuonShowerBxCollection> muonShowers;
      event.getByToken(showerToken, muonShowers);
      for (int bx = muonShowers->getFirstBX(); bx <= muonShowers->getLastBX(); ++bx) {
        if (muonShowers->size(bx) > 0) {
          objMap[bx].shower = muonShowers->at(bx, 0);  // At most one shower per BX.
        }
      }
      return std::make_pair(muonShowers->getFirstBX(), muonShowers->getLastBX());
    }

    std::pair<int, int> MuonPacker::getMuons(GMTOutputObjectMap& objMap,
                                             const edm::Event& event,
                                             const edm::EDGetTokenT<MuonBxCollection>& muonToken) {
      edm::Handle<MuonBxCollection> muons;
      event.getByToken(muonToken, muons);
      for (int bx = muons->getFirstBX(); bx <= muons->getLastBX(); ++bx) {
        objMap[bx] = GMTObjects();
        for (auto mu = muons->begin(bx); mu != muons->end(bx); ++mu) {
          objMap[bx].mus.push_back(*mu);
        }
      }
      return std::make_pair(muons->getFirstBX(), muons->getLastBX());
    }

    void MuonPacker::packBx(const GMTOutputObjectMap& objMap,
                            const int firstMuonBx,
                            const int lastMuonBx,
                            const int firstMuonShowerBx,
                            const int lastMuonShowerBx,
                            PayloadMap& payloadMap) {
      const auto firstBx{std::min(firstMuonShowerBx, firstMuonBx)};
      const auto lastBx{std::max(lastMuonShowerBx, lastMuonBx)};

      for (int bx{firstBx}; bx < lastBx + 1; ++bx) {
        // the first word in every BX and every block id is 0
        for (unsigned int blkId = b1_; blkId < b1_ + 7; blkId += 2) {
          payloadMap[blkId].push_back(0);
        }

        unsigned int blkId = b1_;
        uint32_t mu1_shared_word{0};
        uint32_t mu2_shared_word{0};
        uint32_t mu1_msw{0};
        uint32_t mu2_msw{0};
        uint32_t mu1_lsw{0};
        uint32_t mu2_lsw{0};
        auto mu{objMap.at(bx).mus.begin()};  // Need to get the first muon of that bx from the object map
        std::array<std::array<uint32_t, 4>, 2> const showerWords{
            MuonRawDigiTranslator::getPackedShowerDataWords(objMap.at(bx).shower, fedId_, fwId_)};
        // Slightly convoluted logic to account for the Run-3 muon readout record:
        // To make space for displacement information we moved the raw
        // (i.e. non-extrapolated) eta value to the second "spare" word
        // in the block which we call "shared word". So the logic below
        // needs to be aware if it is operating on the first or second
        // muon in the block in order to place the eta value in the right
        // place in the shared word. Additionally the logic needs to
        // wait for the second muon in the block before filling the
        // payload map because the shared word goes in first.
        for (int muCtr = 1; muCtr <= 8; ++muCtr) {
          if (mu != objMap.at(bx).mus.end()) {
            MuonRawDigiTranslator::generatePackedMuonDataWords(
                *mu, mu2_shared_word, mu2_lsw, mu2_msw, fedId_, fwId_, 2 - (muCtr % 2));
            ++mu;
          }

          // If we're remaining in the current block the muon we just packed is the first one in the block.
          // If not we add both muons to the payload map and go to the next block.
          if (muCtr % 2 == 1) {
            mu1_shared_word = mu2_shared_word;
            mu1_lsw = mu2_lsw;
            mu1_msw = mu2_msw;
            mu1_msw |= showerWords.at(0).at(muCtr / 2);
          } else {
            mu2_msw |= showerWords.at(1).at(muCtr / 2 - 1);
            payloadMap[blkId].push_back(mu1_shared_word | mu2_shared_word);
            payloadMap[blkId].push_back(mu1_lsw);
            payloadMap[blkId].push_back(mu1_msw);
            payloadMap[blkId].push_back(mu2_lsw);
            payloadMap[blkId].push_back(mu2_msw);

            blkId += 2;

            mu1_shared_word = 0;
            mu1_lsw = 0;
            mu1_msw = 0;
          }
          mu2_shared_word = 0;
          mu2_lsw = 0;
          mu2_msw = 0;
        }
      }
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKER(l1t::stage2::GTMuonPacker);
DEFINE_L1T_PACKER(l1t::stage2::GMTMuonPacker);

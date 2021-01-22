#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "GMTTokens.h"
#include "RegionalMuonGMTPacker.h"

namespace l1t {
  namespace stage2 {
    Blocks RegionalMuonGMTPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      auto bmtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenBMTF();
      auto omtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenOMTF();
      auto emtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenEMTF();

      Blocks blocks;

      // pack the muons for each TF in blocks
      packTF(event, bmtfToken, blocks);
      packTF(event, omtfToken, blocks);
      packTF(event, emtfToken, blocks);

      return blocks;
    }

    void RegionalMuonGMTPacker::packTF(const edm::Event& event,
                                       const edm::EDGetTokenT<RegionalMuonCandBxCollection>& tfToken,
                                       Blocks& blocks) {
      edm::Handle<RegionalMuonCandBxCollection> muons;
      event.getByToken(tfToken, muons);

      constexpr unsigned wordsPerBx = 6;  // number of 32 bit words per BX

      PayloadMap payloadMap;

      const auto nBx = muons->getLastBX() - muons->getFirstBX() + 1;
      unsigned bxCtr = 0;
      for (int i = muons->getFirstBX(); i <= muons->getLastBX(); ++i, ++bxCtr) {
        for (auto mu = muons->begin(i); mu != muons->end(i); ++mu) {
          const auto linkTimes2 = mu->link() * 2;

          // If the map key is new reserve the payload size.
          if (payloadMap.count(linkTimes2) == 0) {
            payloadMap[linkTimes2].reserve(wordsPerBx * nBx);
            // If there was no muon on the link of this muon in previous
            // BX the payload up to this BX must be filled with zeros.
            if (bxCtr > 0) {
              while (payloadMap[linkTimes2].size() < bxCtr * wordsPerBx) {
                payloadMap[linkTimes2].push_back(0);
              }
            }
          }

          // Fill the muon in the payload for this link.
          uint32_t msw = 0;
          uint32_t lsw = 0;

          RegionalMuonRawDigiTranslator::generatePackedDataWords(*mu, lsw, msw, isKalman_);

          payloadMap[linkTimes2].push_back(lsw);
          payloadMap[linkTimes2].push_back(msw);
        }

        // padding to 3 muons per block id (link) per BX
        // This can be empty muons as well.
        for (auto& kv : payloadMap) {
          while (kv.second.size() < (bxCtr + 1) * wordsPerBx) {
            kv.second.push_back(0);
          }
        }
      }

      // push everything in the blocks vector
      for (auto& kv : payloadMap) {
        blocks.push_back(Block(kv.first, kv.second));
      }
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKER(l1t::stage2::RegionalMuonGMTPacker);

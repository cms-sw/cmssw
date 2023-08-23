#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "GMTTokens.h"
#include "RegionalMuonGMTPacker.h"

namespace l1t {
  namespace stage2 {
    Blocks RegionalMuonGMTPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      GMTObjectMap bmtfObjMap;
      GMTObjectMap omtfObjMap;
      GMTObjectMap emtfObjMap;
      auto const bmtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenBMTF();
      auto const omtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenOMTF();
      auto const emtfToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonCandTokenEMTF();

      // First we put the muons in our object map.
      std::pair<int, int> const bmtfMuonBx = getMuons(bmtfObjMap, event, bmtfToken);
      std::pair<int, int> const omtfMuonBx = getMuons(omtfObjMap, event, omtfToken);
      std::pair<int, int> const emtfMuonBx = getMuons(emtfObjMap, event, emtfToken);

      // Then the showers (We don't expect to have shower data from BMTF and OMTF -- at the moment, at least)
      std::pair<int, int> emtfMuonShowerBx{0, 0};
      if (useEmtfNominalTightShowers_ || useEmtfLooseShowers_) {
        auto const emtfShowerToken = static_cast<const GMTTokens*>(toks)->getRegionalMuonShowerTokenEMTF();
        emtfMuonShowerBx = getMuonShowers(emtfObjMap, event, emtfShowerToken);
      }

      Blocks blocks;

      // Pack the muons and showers for each TF in blocks
      packTF(bmtfObjMap, bmtfMuonBx.first, bmtfMuonBx.second, 0, 0, blocks);
      packTF(omtfObjMap, omtfMuonBx.first, omtfMuonBx.second, 0, 0, blocks);
      packTF(emtfObjMap, emtfMuonBx.first, emtfMuonBx.second, emtfMuonShowerBx.first, emtfMuonShowerBx.second, blocks);

      return blocks;
    }

    std::pair<int, int> RegionalMuonGMTPacker::getMuonShowers(
        GMTObjectMap& objMap,
        const edm::Event& event,
        const edm::EDGetTokenT<RegionalMuonShowerBxCollection>& tfShowerToken) {
      edm::Handle<RegionalMuonShowerBxCollection> muonShowers;
      event.getByToken(tfShowerToken, muonShowers);
      for (int bx = muonShowers->getFirstBX(); bx <= muonShowers->getLastBX(); ++bx) {
        for (auto muShower = muonShowers->begin(bx); muShower != muonShowers->end(bx); ++muShower) {
          objMap[bx][muShower->link()].shower = *muShower;
        }
      }
      return std::make_pair(muonShowers->getFirstBX(), muonShowers->getLastBX());
    }

    std::pair<int, int> RegionalMuonGMTPacker::getMuons(GMTObjectMap& objMap,
                                                        const edm::Event& event,
                                                        const edm::EDGetTokenT<RegionalMuonCandBxCollection>& tfToken) {
      edm::Handle<RegionalMuonCandBxCollection> muons;
      event.getByToken(tfToken, muons);
      for (int bx = muons->getFirstBX(); bx <= muons->getLastBX(); ++bx) {
        for (auto mu = muons->begin(bx); mu != muons->end(bx); ++mu) {
          objMap[bx][mu->link()].mus.push_back(*mu);
        }
      }
      return std::make_pair(muons->getFirstBX(), muons->getLastBX());
    }

    void RegionalMuonGMTPacker::packTF(const GMTObjectMap& objMap,
                                       const int firstMuonBx,
                                       const int lastMuonBx,
                                       const int firstMuonShowerBx,
                                       const int lastMuonShowerBx,
                                       Blocks& blocks) {
      const auto firstBx{std::min(firstMuonShowerBx, firstMuonBx)};
      const auto lastBx{std::max(lastMuonShowerBx, lastMuonBx)};
      const auto nBx{lastBx - firstBx + 1};

      PayloadMap payloadMap;

      unsigned bxCtr = 0;
      for (int bx{firstBx}; bx < lastBx + 1; ++bx, ++bxCtr) {
        if (objMap.count(bx) > 0) {
          for (const auto& linkMap : objMap.at(bx)) {
            const auto linkTimes2{linkMap.first * 2};
            const auto gmtObjects{linkMap.second};

            // If the payload map key is new reserve the payload size.
            if (payloadMap.count(linkTimes2) == 0) {
              payloadMap[linkTimes2].reserve(wordsPerBx_ * nBx);
              // If there was no muon on the link of this muon in previous
              // BX the payload up to this BX must be filled with zeros.
              if (bxCtr > 0) {
                while (payloadMap[linkTimes2].size() < bxCtr * wordsPerBx_) {
                  payloadMap[linkTimes2].push_back(0);
                }
              }
            }

            if (gmtObjects.mus.size() > 3) {
              edm::LogError("L1T") << "Muon collection for link " << linkMap.first << " has " << gmtObjects.mus.size()
                                   << " entries, but 3 is the maximum!";
            }

            std::array<uint32_t, 6> buf{};  // Making sure contents of buf are initialised to 0.
            size_t frameIdx{0};
            for (const auto& mu : gmtObjects.mus) {
              // Fill the muon in the payload for this link.
              uint32_t msw{0};
              uint32_t lsw{0};

              RegionalMuonRawDigiTranslator::generatePackedDataWords(mu, lsw, msw, isKbmtf_, useEmtfDisplacementInfo_);

              buf.at(frameIdx++) = lsw;
              buf.at(frameIdx++) = msw;
            }
            // Add shower bits to the payload buffer.
            RegionalMuonRawDigiTranslator::generatePackedShowerPayload(
                gmtObjects.shower, buf, useEmtfNominalTightShowers_, useEmtfLooseShowers_);

            payloadMap[linkTimes2].insert(
                payloadMap[linkTimes2].end(), std::move_iterator(buf.begin()), std::move_iterator(buf.end()));
          }
        }
        // We now loop over all channels in the payload map and make sure that they are filled up to the current BX
        // If they are not, we fill them with zeros
        for (auto& kv : payloadMap) {
          while (kv.second.size() < (bxCtr + 1) * wordsPerBx_) {
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

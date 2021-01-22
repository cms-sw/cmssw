#include "BMTFPackerOutput.h"
#include <vector>

// Implementation
namespace l1t {
  namespace stage2 {
    Blocks BMTFPackerOutput::pack(const edm::Event& event, const PackerTokens* toks) {
      int board_id = (int)board();

      auto muonToken = static_cast<const BMTFTokens*>(toks)->getOutputMuonToken();

      Blocks blocks;
      const int bmtfBlockID = 123;
      edm::LogInfo("L1T-BMTFPackerOutput") << "Will use setup:"
                                           << " isKalman->" << isKalman_;

      edm::Handle<RegionalMuonCandBxCollection> muons;
      event.getByToken(muonToken, muons);

      for (auto imu = muons->begin(); imu != muons->end(); imu++) {
        if (imu->processor() + 1 == board_id) {
          uint32_t firstWord(0), lastWord(0);
          RegionalMuonRawDigiTranslator::generatePackedDataWords(*imu, firstWord, lastWord, isKalman_);
          payloadMap_[bmtfBlockID].push_back(firstWord);  //imu->link()*2+1
          payloadMap_[bmtfBlockID].push_back(lastWord);   //imu->link()*2+1
        }
      }  //imu

      //in case less than 3 muons have been found by the processor
      if (payloadMap_[bmtfBlockID].size() < 6) {
        unsigned int initialSize = payloadMap_[bmtfBlockID].size();

        for (unsigned int j = 0; j < 3 - initialSize / 2; j++) {
          payloadMap_[bmtfBlockID].push_back(0);
          uint32_t nullMuon_word2 = 0 | ((65532 & 0xFFFF) << 3) | ((2 & 0x3) << 0);
          payloadMap_[bmtfBlockID].push_back(nullMuon_word2);
        }
      } else if (payloadMap_[bmtfBlockID].size() < 30 && payloadMap_[bmtfBlockID].size() > 6) {
        unsigned int initialSize = payloadMap_[bmtfBlockID].size();

        for (unsigned int j = 0; j < 15 - initialSize / 2; j++) {
          payloadMap_[bmtfBlockID].push_back(0);
          uint32_t nullMuon_word2 = 0 | ((65532 & 0xFFFF) << 3) | ((2 & 0x3) << 0);
          payloadMap_[bmtfBlockID].push_back(nullMuon_word2);
        }
      }

      Block block(bmtfBlockID, payloadMap_[bmtfBlockID]);

      blocks.push_back(block);

      return blocks;
    }

  }  // namespace stage2
}  // namespace l1t
DEFINE_L1T_PACKER(l1t::stage2::BMTFPackerOutput);

#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "GTTokens.h"

#include "CaloSummaryPacker.h"

namespace l1t {
  namespace stage2 {
    std::vector<uint32_t> CaloSummaryPacker::generateCICADAWordsFromScore(float score) {
      //Shift the score up by 8 bits, and turn it into an integer for easy working with
      //the bits
      uint32_t cicadaBits = static_cast<uint32_t>(score * 256.f);
      //CICADA word goes from most significant bits, to least significant
      //But always at the start of the word, so the resultant words need to be adjusted upwards
      uint32_t firstWord = (cicadaBits & 0xF000) << 16;
      uint32_t secondWord = (cicadaBits & 0x0F00) << 20;
      uint32_t thirdWord = (cicadaBits & 0x00F0) << 24;
      uint32_t fourthWord = (cicadaBits & 0x000F) << 28;

      return {firstWord, secondWord, thirdWord, fourthWord};
    }

    Blocks CaloSummaryPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      edm::Handle<CICADABxCollection> cicadaScores;
      event.getByToken(static_cast<const GTTokens*>(toks)->getCICADAToken(), cicadaScores);

      std::vector<uint32_t> payload;

      for (int i = cicadaScores->getFirstBX(); i <= cicadaScores->getLastBX(); ++i) {
        //check if we _have_ a CICADA score to work with, the emulator does not
        //guarantee 5 BX worth of CICADA, but unpacked data from uGT does
        //If we do not have a CICADA score for a BX, we can simply treat that as
        //a zero score.
        //otherwise, once we have a CICADA score, we can simply construct the 6
        //words (4 with CICADA bits, 2 without), from the score
        float CICADAScore = 0.0;
        if (cicadaScores->size(i) != 0) {
          CICADAScore = cicadaScores->at(
              i, 0);  //Shouldn't ever have more than one score per BX, so this should be safe if the size is not 0
        }
        //Make the CICADA words
        std::vector<uint32_t> bxWords = generateCICADAWordsFromScore(CICADAScore);
        payload.insert(payload.end(), bxWords.begin(), bxWords.end());
        //Remaining two 32 bit words of CICADA are unused
        payload.push_back(0);
        payload.push_back(0);
      }

      return {Block(22, payload)};
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKER(l1t::stage2::CaloSummaryPacker);

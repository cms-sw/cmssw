#include "FWCore/Framework/interface/Event.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "CaloLayer1Tokens.h"

#include "CICADAPacker.h"

namespace l1t {
  namespace stage2 {
    std::vector<uint32_t> CICADAPacker::makeCICADAWordsFromScore(float score) {
      //turn the score into an integer to make it a bit easier to work with it bitwise
      uint32_t cicadaBits = static_cast<uint32_t>(score * 256.f);
      uint32_t firstWord = (cicadaBits & 0xF000) << 16;
      uint32_t secondWord = (cicadaBits & 0x0F00) << 20;
      uint32_t thirdWord = (cicadaBits & 0x00F0) << 24;
      uint32_t fourthWord = (cicadaBits & 0x000F) << 28;
      return {firstWord, secondWord, thirdWord, fourthWord};
    }

    Blocks CICADAPacker::pack(const edm::Event& event, const PackerTokens* toks) {
      edm::Handle<CICADABxCollection> cicadaScores;
      event.getByToken(static_cast<const CaloLayer1Tokens*>(toks)->getCICADAToken(), cicadaScores);

      std::vector<uint32_t> payload;

      //for the purposes of this packer, we really only have to care about the central BX
      //Calo Layer 1 doesn't distinguish between BX's, it just sends the one.
      float cicadaScore = 0.0;
      if (cicadaScores->size(0) != 0) {
        cicadaScore = cicadaScores->at(0, 0);
      }
      payload = makeCICADAWordsFromScore(cicadaScore);
      payload.push_back(0);
      payload.push_back(0);

      return {Block(0, payload, 0, 0, CTP7)};
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKER(l1t::stage2::CICADAPacker);

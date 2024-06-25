#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CICADAUnpacker.h"

#include <cmath>

using namespace edm;

namespace l1t {
  namespace stage2 {
    bool CICADAUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      LogDebug("L1T") << "Block Size = " << block.header().getSize();
      LogDebug("L1T") << "Board ID = " << block.amc().getBoardID();

      auto res = static_cast<CaloLayer1Collections*>(coll)->getCICADABxCollection();
      // default BX range to trigger standard -2 to 2
      // Even though CICADA will never have BX information
      // And everything gets put in BX 0
      res->setBXRange(-2, 2);

      int amc_slot = block.amc().getAMCNumber();
      if (not(amc_slot == 7)) {
        throw cms::Exception("CICADAUnpacker")
            << "Calo Summary (CICADA) unpacker is unpacking an unexpected AMC. Expected AMC number 7, got AMC number "
            << amc_slot << std::endl;
        return false;
      } else {
        std::vector<uint32_t> cicadaWords = {0, 0, 0, 0};
        //the last 4 words are CICADA words
        for (uint32_t i = 2; i < 6; ++i) {
          cicadaWords.at(i - 2) = ((block.payload().at(i)) >> 28);
        }

        float cicadaScore = convertCICADABitsToFloat(cicadaWords);
        res->push_back(0, cicadaScore);
        return true;
      }
    }

    //convert the 4 CICADA bits/words into a proper number
    float CICADAUnpacker::convertCICADABitsToFloat(const std::vector<uint32_t>& cicadaBits) {
      uint32_t tempResult = 0;
      tempResult |= cicadaBits.at(0) << 12;
      tempResult |= cicadaBits.at(1) << 8;
      tempResult |= cicadaBits.at(2) << 4;
      tempResult |= cicadaBits.at(3);
      float result = 0.0;
      result = (float)tempResult * pow(2.0, -8);
      return result;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::CICADAUnpacker);

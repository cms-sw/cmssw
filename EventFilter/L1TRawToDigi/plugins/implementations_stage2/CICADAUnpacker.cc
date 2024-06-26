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
        const uint32_t* base = block.payload().data();
        //This differs slightly from uGT, in that we grab the first 4 bits
        //of the last 4 words (still in first to last order) and arrange those
        uint32_t word = (caloCrateCicadaBitsPattern & base[2]) >> 16 | (caloCrateCicadaBitsPattern & base[3]) >> 20 |
                        (caloCrateCicadaBitsPattern & base[4]) >> 24 | (caloCrateCicadaBitsPattern & base[5]) >> 28;
        float score = static_cast<float>(word) / 256.f;
        res->push_back(0, score);
        return true;
      }
    }

  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::CICADAUnpacker);

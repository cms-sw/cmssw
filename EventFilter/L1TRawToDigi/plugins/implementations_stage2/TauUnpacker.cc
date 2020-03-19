#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "L1TStage2Layer2Constants.h"
#include "TauUnpacker.h"

namespace l1t {
  namespace stage2 {
    TauUnpacker::TauUnpacker() : TauCopy_(0) {}

    bool TauUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      using namespace l1t::stage2::layer2;

      LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

      int nBX = int(ceil(block.header().getSize() / demux::nOutputFramePerBX));

      // Find the first and last BXs
      int firstBX = -(ceil((double)nBX / 2.) - 1);
      int lastBX;
      if (nBX % 2 == 0) {
        lastBX = ceil((double)nBX / 2.);
      } else {
        lastBX = ceil((double)nBX / 2.) - 1;
      }

      auto res_ = static_cast<L1TObjectCollections*>(coll)->getTaus(TauCopy_);
      res_->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Loop over multiple BX and then number of Tau cands filling collection
      for (int bx = firstBX; bx <= lastBX; bx++) {
        for (unsigned iTau = 0; iTau < demux::nTauPerLink && iTau < block.header().getSize(); iTau++) {
          int iFrame = (bx - firstBX) * demux::nOutputFramePerBX + iTau;

          uint32_t raw_data = block.payload().at(iFrame);

          if (raw_data == 0)
            continue;

          l1t::Tau tau = l1t::Tau();

          tau.setHwPt(raw_data & 0x1FF);

          if (tau.hwPt() == 0)
            continue;

          int abs_eta = (raw_data >> 9) & 0x7F;
          if ((raw_data >> 16) & 0x1) {
            tau.setHwEta(-1 * (128 - abs_eta));
          } else {
            tau.setHwEta(abs_eta);
          }

          tau.setHwPhi((raw_data >> 17) & 0xFF);
          tau.setHwIso((raw_data >> 25) & 0x3);   // 2 bits
          tau.setHwQual((raw_data >> 27) & 0x7);  // Assume 3 bits for now? leaves 3 spare bits
          // tau.setHwQual((raw_data >> 26) & 0x7); // Assume 3 bits for now? leaves 3 spare bits

          LogDebug("L1T") << "Tau: eta " << tau.hwEta() << " phi " << tau.hwPhi() << " pT " << tau.hwPt() << " iso "
                          << tau.hwIso() << " qual " << tau.hwQual();

          tau.setP4(l1t::CaloTools::p4Demux(&tau));

          res_->push_back(bx, tau);
        }
      }

      return true;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::TauUnpacker);

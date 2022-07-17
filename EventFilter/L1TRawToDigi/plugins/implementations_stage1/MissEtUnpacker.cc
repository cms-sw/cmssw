#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CaloCollections.h"
#include "MissEtUnpacker.h"

namespace l1t {
  namespace stage1 {
    bool MissEtUnpacker::unpack(const Block& block, UnpackerCollections* coll) {
      LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

      int nBX, firstBX, lastBX;
      nBX = int(ceil(block.header().getSize() / 2.));
      getBXRange(nBX, firstBX, lastBX);

      auto res_ = static_cast<CaloCollections*>(coll)->getEtSums();
      res_->setBXRange(firstBX, lastBX);

      LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

      // Initialise index
      int unsigned i = 0;

      // Loop over multiple BX and then number of jets filling jet collection
      for (int bx = firstBX; bx <= lastBX; bx++) {
        res_->resize(bx, 4);

        uint32_t raw_data0 = block.payload()[i++];
        uint32_t raw_data1 = block.payload()[i++];

        /* if (raw_data0 == 0 || raw_data1==0) continue; */

        uint16_t candbit[2];
        candbit[0] = raw_data0 & 0xFFFF;
        candbit[1] = raw_data1 & 0xFFFF;

        int etmiss = candbit[0] & 0xFFF;
        int overflowetmiss = (candbit[0] >> 12) & 0x1;
        int etmissphi = candbit[1] & 0x7F;

        l1t::EtSum met{l1t::EtSum::kMissingEt};
        met.setHwPt(etmiss);
        met.setHwPhi(etmissphi);
        int flagetmiss = met.hwQual();
        flagetmiss |= overflowetmiss;
        met.setHwQual(flagetmiss);
        LogDebug("L1T") << "MET: pT " << met.hwPt() << "is overflow " << overflowetmiss << std::endl;
        res_->set(bx, 0, met);
      }

      return true;
    }
  }  // namespace stage1
}  // namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage1::MissEtUnpacker);

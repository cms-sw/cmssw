#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
  namespace stage1 {
    class MissHtUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    bool
      MissHtUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {

        LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

        int nBX, firstBX, lastBX;
        nBX = int(ceil(block.header().getSize() / 2.)); 
        getBXRange(nBX, firstBX, lastBX);

        auto reset_ = static_cast<CaloCollections*>(coll)->getEtSums();
        reset_->setBXRange(firstBX, lastBX);

        LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

        // Initialise index
        int unsigned i = 0;

        // Loop over multiple BX and then number of jets filling jet collection
        for (int bx=firstBX; bx<=lastBX; bx++){

          reset_->resize(bx,4);

          uint32_t raw_data0 = block.payload()[i++];
          uint32_t raw_data1 = block.payload()[i++];        

          /* if (raw_data0 == 0 || raw_data1==0) continue; */

          uint16_t candbit[2];
          candbit[0] = raw_data0 & 0xFFFF;
          candbit[1] = raw_data1 & 0xFFFF;

          int htmissphi=candbit[0] & 0x1F;
          int htmiss=(candbit[0]>>5) & 0x7F;
          int overflowhtmiss=(candbit[0]>>12) & 0x1;

          l1t::EtSum mht = l1t::EtSum();
          mht.setHwPt(htmiss);
          mht.setHwPhi(htmissphi);
          mht.setType(l1t::EtSum::kMissingHt); 
          int flaghtmiss=mht.hwQual();
          flaghtmiss|= overflowhtmiss;
          mht.setHwQual(flaghtmiss);       
          LogDebug("L1T") << "MHT: pT " << mht.hwPt()<<"is overflow "<<overflowhtmiss<<std::endl;
          reset_->set(bx, 1,mht);

        }

        return true;

      }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::MissHtUnpacker);

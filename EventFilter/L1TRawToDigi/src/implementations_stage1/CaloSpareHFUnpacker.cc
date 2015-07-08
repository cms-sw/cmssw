#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
  namespace stage1 {
    class CaloSpareHFUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    bool
      CaloSpareHFUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {

        LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

        int nBX, firstBX, lastBX;
        nBX = int(ceil(block.header().getSize() / 2.)); 
        getBXRange(nBX, firstBX, lastBX);

        auto resHFBitCounts_ = static_cast<CaloCollections*>(coll)->getCaloSpareHFBitCounts();
        resHFBitCounts_->setBXRange(firstBX, lastBX);

        auto resHFRingSums_ = static_cast<CaloCollections*>(coll)->getCaloSpareHFRingSums();
        resHFRingSums_->setBXRange(firstBX, lastBX);
        
        LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

        // Initialise index
        int unsigned i = 0;

        // Loop over multiple BX and then number of jets filling jet collection
        for (int bx=firstBX; bx<=lastBX; bx++){
          uint32_t raw_data0 = block.payload()[i++];
          uint32_t raw_data1 = block.payload()[i++];        

          /* if (raw_data0 == 0 || raw_data1==0) continue; */

          uint16_t candbit[2];
          candbit[0] = raw_data0 & 0xFFFF;
          candbit[1] = raw_data1 & 0xFFFF;

          int hfbitcount=candbit[0] & 0xFFF;
          int hfringsum=((candbit[0]>>12) & 0x7) | ((candbit[1] & 0x1FF) << 3);
          
          l1t::CaloSpare hfbc= l1t::CaloSpare();
          hfbc.setHwPt(hfbitcount);
          hfbc.setType(l1t::CaloSpare::HFBitCount);  
          LogDebug("L1T") << "hfbc pT " << hfbc.hwPt(); 
          resHFBitCounts_->push_back(bx,hfbc);        
          
          l1t::CaloSpare hfrs= l1t::CaloSpare();
          hfrs.setHwPt(hfringsum);
          hfrs.setType(l1t::CaloSpare::HFRingSum);  
          LogDebug("L1T") << "hfrs pT " << hfrs.hwPt();  
          resHFRingSums_->push_back(bx,hfrs);       

        }

        return true;

      }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::CaloSpareHFUnpacker);

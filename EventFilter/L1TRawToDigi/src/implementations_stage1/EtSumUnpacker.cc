#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
  namespace stage1 {
    class EtSumUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    bool
      EtSumUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {

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
        for (int bx=firstBX; bx<=lastBX; bx++){

          res_->resize(bx,4);

          uint32_t raw_data0 = block.payload()[i++];
          uint32_t raw_data1 = block.payload()[i++];        

          /* if (raw_data0 == 0 || raw_data1==0) continue; */

          uint16_t candbit[2];
          candbit[0] = raw_data0 & 0xFFFF;
          candbit[1] = raw_data1 & 0xFFFF;

          int totet=candbit[0] & 0xFFF;
          int overflowtotet=(candbit[0]>>12) & 0x1;
          int totht=candbit[1] & 0xFFF;
          int overflowtotht=(candbit[1]>>12) & 0x1;

          l1t::EtSum et = l1t::EtSum();
          et.setHwPt(totet);
          et.setType(l1t::EtSum::kTotalEt);      
          int flagtotet=et.hwQual();
          flagtotet|= overflowtotet;
          et.setHwQual(flagtotet);       
          LogDebug("L1T") << "ET: pT " << et.hwPt()<<"is overflow "<<overflowtotet<<std::endl;
          res_->set(bx, 2,et);

          l1t::EtSum ht = l1t::EtSum();
          ht.setHwPt(totht);
          ht.setType(l1t::EtSum::kTotalHt);       
          int flagtotht=ht.hwQual();
          flagtotht|= overflowtotht;
          ht.setHwQual(flagtotht);       
          LogDebug("L1T") << "HT: pT " << ht.hwPt()<<"is overflow "<<overflowtotht<<std::endl;
          res_->set(bx, 3,ht);

        }

        return true;

      }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::EtSumUnpacker);

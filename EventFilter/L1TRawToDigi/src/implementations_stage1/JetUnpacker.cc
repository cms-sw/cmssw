#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
  namespace stage1 {
    class JetUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

// Implementation

namespace l1t {
  namespace stage1 {
    bool
      JetUnpacker::unpack(const Block& block, UnpackerCollections *coll)
      {

        LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

        int nBX = int(ceil(block.header().getSize() / 12.)); // Since there are 12 jets reported per event (see CMS IN-2013/005)

        // Find the first and last BXs
        int firstBX = -(ceil((double)nBX/2.)-1);
        int lastBX;
        if (nBX % 2 == 0) {
          lastBX = ceil((double)nBX/2.)+1;
        } else {
          lastBX = ceil((double)nBX/2.);
        }

        auto res_ = static_cast<CaloCollections*>(coll)->getJets();
        res_->setBXRange(firstBX, lastBX);

        LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

        // Initialise index
        int unsigned i = 0;

        // Loop over multiple BX and then number of jets filling jet collection
        for (int bx=firstBX; bx<lastBX; bx++){
          uint32_t raw_data0 = block.payload()[i++];
          uint32_t raw_data1 = block.payload()[i++];        

          /* if (raw_data0 == 0 || raw_data1==0) continue; */

          uint16_t jetbit[4];
          jetbit[0] = raw_data0 & 0xFFFF;
          jetbit[1] = (raw_data0 >> 16) & 0xFFFF;
          jetbit[2] = raw_data1 & 0xFFFF;
          jetbit[3] = (raw_data1 >> 16) & 0xFFFF;

          int jetPt;
          int jetEta;
          int jetEtasign;
          int jetPhi;
          int jetqualflag;

          for (int ijet=0;ijet<4;ijet++){

            jetPt=jetbit[ijet] & 0x3F;
            jetEta=(jetbit[ijet]>>6 ) & 0x7;
            jetEtasign=(jetbit[ijet]>>9) & 0x1;
            jetPhi=(jetbit[ijet]>>10) & 0x1F;
            jetqualflag=(jetbit[ijet]>>15) & 0x1;

            l1t::Jet jet = l1t::Jet();
            jet.setHwPt(jetPt);
            if(jetEtasign) jet.setHwEta(-1 * jetEta);
            else jet.setHwEta(jetEta);
            jet.setHwPhi(jetPhi);
            jet.setHwQual(jetqualflag);

            std::cout << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual() << std::endl;
            res_->push_back(bx,jet);
          }        
        }

        return true;
      }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::JetUnpacker);

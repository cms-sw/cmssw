#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage2 {
      class CaloTowerUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   bool
   CaloTowerUnpacker::unpack(const Block& block, UnpackerCollections *coll)
   {

     // check this is the correct MP
     unsigned int amc  = block.amc().getAMCNumber();
     unsigned int bxid = block.amc().getBX();
     //     if( (amc-1) != ((bxid-1)%9) ) return true;
     if( (amc-1) != ((bxid-1+3)%9) ) return true;    // temporary measure !
     LogDebug("L1T") << "Unpacking AMC " << amc << " for BX " << bxid;

     // Link number is block_ID / 2
     unsigned link = block.header().getID()/2;
     
     // Also need link number rounded down to even number
     unsigned link_phi = (link % 2 == 0) ? link : (link -1);

     // number of frames used in a few different places
     unsigned nframes=40;

     int nBX = int(ceil(block.header().getSize()/nframes)); // Since there are two Rx links per block with 2*28 slices in barrel and endcap + 2*13 for upgraded HF 

     // Find the first and last BXs
     int firstBX = -(std::ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = std::ceil((double)nBX/2.);
     } else {
       lastBX = std::ceil((double)nBX/2.)-1;
     }

     auto res_ = static_cast<CaloCollections*>(coll)->getTowers();
     res_->setBXRange(std::min(firstBX, res_->getFirstBX()), std::max(lastBX, res_->getLastBX()));

     LogDebug("L1T") << "Block : id=" << block.header().getID() << ", size=" << block.header().getSize() << ", link=" << link << ", link_phi=" << link_phi << ", nBX=" << nBX << ", firstBX=" << firstBX << ", lastBX=" << lastBX;

     // Loop over multiple BX and fill towers collection
     for (int bx=firstBX; bx<=lastBX; bx++){

       for (unsigned iframe=0; iframe<nframes && iframe<block.header().getSize(); ++iframe) {

         uint32_t raw_data = block.payload().at(iframe);

         if ((raw_data & 0xFFFF) != 0) {

           l1t::CaloTower tower1 = l1t::CaloTower();
    
           // First calo tower is in the LSW with phi
           tower1.setHwPt(raw_data & 0x1FF);
           tower1.setHwQual((raw_data >> 12) & 0xF);
           tower1.setHwEtRatio((raw_data >>9) & 0x7);
           tower1.setHwPhi(link_phi+1); // iPhi starts at 1

	   int ieta=iframe+1;
           if (link % 2!=0) ieta = ieta * -1;

	   tower1.setHwEta( CaloTools::caloEta(ieta) );
	 
           LogDebug("L1T") << "Tower 1: Eta " << tower1.hwEta() 
                           << " phi " << tower1.hwPhi() 
                           << " pT " << tower1.hwPt() 
                           << " frame " << iframe 
                           << " qual " << tower1.hwQual() 
                           << " EtRatio " << tower1.hwEtRatio();

           res_->push_back(bx,tower1);
         }

         if (((raw_data >> 16)& 0xFFFF) != 0) {

           // Second calo tower is in the MSW with phi+1
           l1t::CaloTower tower2 = l1t::CaloTower();
	 
           tower2.setHwPt((raw_data >> 16) & 0x1FF);
           tower2.setHwQual((raw_data >> 28 ) & 0xF);
           tower2.setHwEtRatio((raw_data >> 25) & 0x7);
           tower2.setHwPhi(link_phi+2);

	   int ieta = iframe+1;
           if (link % 2!=0) ieta = ieta * -1;
	   tower2.setHwEta( CaloTools::caloEta(ieta) );
	 
           LogDebug("L1T") << "Tower 2: Eta " << tower2.hwEta()
                           << " phi " << tower2.hwPhi()
                           << " pT " << tower2.hwPt()
                           << " frame " << iframe
                           << " qual " << tower2.hwQual()
                           << " EtRatio " << tower2.hwEtRatio();

           res_->push_back(bx,tower2);
	 }
       }
     }
     
     return true;

  }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::CaloTowerUnpacker);

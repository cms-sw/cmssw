#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1TObjectCollections.h"
//#include "GTCollections.h"

namespace l1t {
   namespace stage2 {
      class MuonUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   bool
   MuonUnpacker::unpack(const Block& block, UnpackerCollections *coll)
   {

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

     int nBX = int(ceil(block.header().getSize() / 12.)); // Since there are 12 EGamma objects reported per event (see CMS IN-2013/005)

     // Find the central, first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.);
     } else {
       lastBX = ceil((double)nBX/2.)-1;
     }

     auto res_ = static_cast<L1TObjectCollections*>(coll)->getMuons();
     res_->setBXRange(firstBX, lastBX);

     LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

     // Initialise index
     int unsigned i = 0;

     // Loop over multiple BX and then number of EG cands filling collection
     for (int bx=firstBX; bx<=lastBX; bx++){
       
       for (unsigned nMu=0; nMu < block.header().getSize(); nMu=nMu+2){ //need to check the 12*2...this is not really num muons it is max words in block.

         //muons are spread over 64 bits grab the two 32 bit pieces
         uint32_t raw_data_00_31 = block.payload()[i++];
	 uint32_t raw_data_32_63 = block.payload()[i++];

         std::cout << "nMu = " << nMu << "Word 1 " << hex << raw_data_00_31 << " Word 2 " << raw_data_32_63 << dec << std::endl; 
	 
         // skip padding 
         if (raw_data_00_31 == 0)
            continue;

         l1t::Muon mu = l1t::Muon();

             
         mu.setHwPt( (raw_data_00_31 >> 10) & 0x1FF);
         mu.setHwQual( (raw_data_00_31 >> 19) & 0xF); 

	 int abs_eta = (raw_data_00_31 >> 23) & 0xFF;
         if ((raw_data_00_31 >> 31) & 0x1) {
           mu.setHwEta(-1*abs_eta);
         } else {
           mu.setHwEta(abs_eta);
         }

         mu.setHwPhi((raw_data_00_31 >> 0) & 0x1FF);
	 mu.setHwIso((raw_data_32_63 >> 0) & 0x3); 
         mu.setHwCharge( (raw_data_32_63 >> 2) & 0x1);
	 mu.setHwChargeValid( (raw_data_32_63 >> 3) & 0x1);
       
         LogDebug("L1T") << "Mu: eta " << mu.hwEta() << " phi " << mu.hwPhi() << " pT " << mu.hwPt() << " iso " << mu.hwIso() << " qual " << mu.hwQual();

         res_->push_back(bx,mu);
       }

     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::MuonUnpacker);

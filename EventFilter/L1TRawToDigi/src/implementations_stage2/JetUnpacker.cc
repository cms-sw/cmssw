#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage2 {
      class JetUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
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
       for (unsigned nJet=0; nJet < 12 && nJet < block.header().getSize(); nJet++){
         uint32_t raw_data = block.payload()[i++];

         if (raw_data == 0)
            continue;

         l1t::Jet jet = l1t::Jet();

         jet.setHwPt(raw_data & 0x7FF);
         
         int abs_eta = (raw_data >> 11) & 0x7F;
         if ((raw_data >> 18) & 0x1) {
           jet.setHwEta(-1 * abs_eta);
         } else {
           jet.setHwEta(abs_eta);
         }

	 jet.setHwPhi((raw_data >> 19) & 0xFF);
         jet.setHwQual((raw_data >> 27) & 0x7); // Assume 3 bits for now? Leaves 2 bits spare

         LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

         res_->push_back(bx,jet);
       }
     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::JetUnpacker);

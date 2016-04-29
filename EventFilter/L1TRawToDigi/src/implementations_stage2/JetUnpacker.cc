#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "L1TStage2Layer2Constants.h"

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

     using namespace l1t::stage2::layer2;

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

     int nBX = int(ceil(block.header().getSize() / (double) demux::nOutputFramePerBX )); // 6 frames per BX

     // Find the first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.);
     } else {
       lastBX = ceil((double)nBX/2.)-1;
     }

     auto res_ = static_cast<L1TObjectCollections*>(coll)->getJets();
     res_->setBXRange(firstBX, lastBX);

     LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

     // Loop over multiple BX and then number of jets filling jet collection
     for (int bx=firstBX; bx<=lastBX; bx++){
       for (unsigned iJet=0; iJet < demux::nJetPerLink && iJet < block.header().getSize(); iJet++){

	 int iFrame = (bx-firstBX)*demux::nOutputFramePerBX + iJet;
         uint32_t raw_data = block.payload().at(iFrame);

         if (raw_data == 0)
            continue;

         l1t::Jet jet = l1t::Jet();

         jet.setHwPt(raw_data & 0x7FF);

	 if (jet.hwPt()==0) continue;
         
         int abs_eta = (raw_data >> 11) & 0x7F;
         if ((raw_data >> 18) & 0x1) {
           jet.setHwEta(-1 * (128-abs_eta));
         } else {
           jet.setHwEta(abs_eta);
         }

	 jet.setHwPhi((raw_data >> 19) & 0xFF);
         jet.setHwQual((raw_data >> 27) & 0x7); // Assume 3 bits for now? Leaves 2 bits spare

         LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual() << " bx " << bx;

	 jet.setP4( l1t::CaloTools::p4Demux(&jet) );

         res_->push_back(bx, jet);
       }
     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::JetUnpacker);

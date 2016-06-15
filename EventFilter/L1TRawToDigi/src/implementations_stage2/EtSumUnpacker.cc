#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "L1TObjectCollections.h"

#include "L1TStage2Layer2Constants.h"

namespace l1t {
   namespace stage2 {
      class EtSumUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   bool
   EtSumUnpacker::unpack(const Block& block, UnpackerCollections *coll)
   {

     using namespace l1t::stage2::layer2;

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

     int nBX = int(ceil(block.header().getSize() / demux::nOutputFramePerBX)); // Since there 6 frames per demux output event
     // expect the first four frames to be the first 4 EtSum objects reported per event (see CMS IN-2013/005)

     // Find the central, first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.);
     } else {
       lastBX = ceil((double)nBX/2.)-1;
     }

     auto res_ = static_cast<L1TObjectCollections*>(coll)->getEtSums();
     res_->setBXRange(firstBX, lastBX);

     LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

     // Loop over multiple BX and fill EtSums collection
     for (int bx=firstBX; bx<=lastBX; bx++){

       // ET
       int iFrame = (bx-firstBX)*demux::nOutputFramePerBX;

       uint32_t raw_data = block.payload().at(iFrame);

       l1t::EtSum et = l1t::EtSum();
    
       et.setHwPt(raw_data & 0xFFF);
       et.setType(l1t::EtSum::kTotalEt);       
       et.setP4( l1t::CaloTools::p4Demux(&et) );

       LogDebug("L1T") << "ET: pT " << et.hwPt() << " bx " << bx;

       res_->push_back(bx,et);

       l1t::EtSum mbp0 = l1t::EtSum();
       mbp0.setHwPt( (raw_data>>28) & 0xf );
       mbp0.setType( l1t::EtSum::kMinBiasHFP0 );

       res_->push_back(bx, mbp0);

       // HT

       raw_data = block.payload()[iFrame+1];

       l1t::EtSum ht = l1t::EtSum();
    
       ht.setHwPt(raw_data & 0xFFF);
       ht.setType(l1t::EtSum::kTotalHt);       
       ht.setP4( l1t::CaloTools::p4Demux(&ht) );

       LogDebug("L1T") << "HT: pT " << ht.hwPt();

       res_->push_back(bx,ht);

       l1t::EtSum mbm0 = l1t::EtSum();
       mbm0.setHwPt( (raw_data>>28) & 0xf );
       mbm0.setType( l1t::EtSum::kMinBiasHFM0 );

       res_->push_back(bx, mbm0);

       //  MET

       raw_data = block.payload()[iFrame+2];

       l1t::EtSum met = l1t::EtSum();
    
       met.setHwPt(raw_data & 0xFFF);
       met.setHwPhi((raw_data >> 12) & 0xFF);
       met.setType(l1t::EtSum::kMissingEt);       
       met.setP4( l1t::CaloTools::p4Demux(&met) );

       LogDebug("L1T") << "MET: phi " << met.hwPhi() << " pT " << met.hwPt() << " bx " << bx;

       res_->push_back(bx,met);

       l1t::EtSum mbp1 = l1t::EtSum();
       mbp1.setHwPt( (raw_data>>28) & 0xf );
       mbp1.setType( l1t::EtSum::kMinBiasHFP1 );

       res_->push_back(bx, mbp1);

       // MHT

       raw_data = block.payload()[iFrame+3];

       l1t::EtSum mht = l1t::EtSum();
    
       mht.setHwPt(raw_data & 0xFFF);
       mht.setHwPhi((raw_data >> 12) & 0xFF);
       mht.setType(l1t::EtSum::kMissingHt);       
       mht.setP4( l1t::CaloTools::p4Demux(&mht) );

       LogDebug("L1T") << "MHT: phi " << mht.hwPhi() << " pT " << mht.hwPt() << " bx " << bx;

       res_->push_back(bx,mht);

       l1t::EtSum mbm1 = l1t::EtSum();
       mbm1.setHwPt( (raw_data>>28) & 0xf );
       mbm1.setType( l1t::EtSum::kMinBiasHFM1 );

       res_->push_back(bx, mbm1);

     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::EtSumUnpacker);

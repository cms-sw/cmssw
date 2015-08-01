#define EDM_ML_DEBUG 1

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage2 {
      class MPUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   bool
   MPUnpacker::unpack(const Block& block, UnpackerCollections *coll)
   {

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

     auto res1_ = static_cast<CaloCollections*>(coll)->getMPJets();
     auto res2_ = static_cast<CaloCollections*>(coll)->getMPEtSums();
     res1_->setBXRange(0,0);
     res2_->setBXRange(0,0);

     // Initialise frame indices for each data type
     int unsigned fet = 0;
     int unsigned fht = 1;
     int unsigned fjet = 2;

     // ET / MET(x) / MET (y)

     uint32_t raw_data = block.payload()[fet];

     l1t::EtSum et = l1t::EtSum();

     et.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
     //et.setHwPt(raw_data & 0xFFFFF);
     switch(block.header().getID()){
     case 1:  et.setType(l1t::EtSum::kTotalEt);  break;
     case 3:  et.setType(l1t::EtSum::kTotalEtx); break;
     case 5:  et.setType(l1t::EtSum::kTotalEty); break;
     case 7:  et.setType(l1t::EtSum::kTotalEt);  break;
     case 9:  et.setType(l1t::EtSum::kTotalEtx); break;
     case 11: et.setType(l1t::EtSum::kTotalEty); break;
     default: break;
     }


     LogDebug("L1T") << "ET/METx/METy: pT " << et.hwPt();

     res2_->push_back(0,et);


     // HT / MHT(x)/ MHT (y)

     raw_data = block.payload()[fht];

     l1t::EtSum ht = l1t::EtSum(); 

     //ht.setHwPt(raw_data & 0xFFFFF);
     ht.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
     switch(block.header().getID()){
     case 1:  ht.setType(l1t::EtSum::kTotalHt);  break;
     case 3:  ht.setType(l1t::EtSum::kTotalHtx); break;
     case 5:  ht.setType(l1t::EtSum::kTotalHty); break;
     case 7:  ht.setType(l1t::EtSum::kTotalHt);  break;
     case 9:  ht.setType(l1t::EtSum::kTotalHtx); break;
     case 11: ht.setType(l1t::EtSum::kTotalHty); break;
     default: break;
     }

     LogDebug("L1T") << "HT/MHTx/MHTy: pT " << ht.hwPt();

     res2_->push_back(0,ht);

     // Two jets
     for (unsigned nJet=0; nJet < 2; nJet++){

       raw_data = block.payload()[fjet+nJet];

       if (raw_data == 0)
            continue;

       l1t::Jet jet = l1t::Jet();

       int etasign = 1;
       if ((block.header().getID() == 7) ||
           (block.header().getID() == 9) ||
           (block.header().getID() == 11)) {
         etasign = -1;
       }

       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;

       jet.setHwEta(etasign*(raw_data & 0x3F));
       jet.setHwPhi((raw_data >> 6) & 0x7F);
       jet.setHwPt((raw_data >> 13) & 0xFFFF);
         
       LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

       //res1_->push_back(0,jet);

       // Push them back in the right place (for checking sorting)

       int blockID = block.header().getID();
       int nPos=0, nNeg=0;
       for (unsigned i=0; i<res1_->size(0); i++)
         res1_->at(0,i).hwEta()>0 ? nPos++ : nNeg++;

       if (nJet==1) res1_->push_back(0,jet);
       else if (nJet==0) {
         if (blockID==1) {
           res1_->push_back(0,jet);
         }
         if (blockID==3) {
           if (nPos==1) res1_->push_back(0,jet);
           else if (nPos==2) res1_->insert(0,1,jet);
         }
         if (blockID==5) {
           if (nPos==2) res1_->push_back(0,jet);
           else if (nPos>2) res1_->insert(0,2,jet);
         }
         if (blockID==7) {
           res1_->push_back(0,jet);
         }
         if (blockID==9) {
           if (nNeg==1) res1_->push_back(0,jet);
           else if (nNeg==2) res1_->insert(0,nPos+1,jet);
         }
         if (blockID==11) {
           if (nNeg==2) res1_->push_back(0,jet);
           else if (nNeg>2) res1_->insert(0,nPos+2,jet);
         }
       }

     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker);

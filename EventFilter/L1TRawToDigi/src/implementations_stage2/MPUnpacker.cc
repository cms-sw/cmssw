#define EDM_ML_DEBUG 1

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

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

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize() << " AMC = " << block.amc().getAMCNumber();

     // check this is the correct MP
     unsigned int amc  = block.amc().getAMCNumber();
     unsigned int bxid = block.amc().getBX();
     //     if( (amc-1) != (bxid-1)%9 ) return true;
     if( (amc-1) != ((bxid-1+3)%9) ) return true;   // temporary measure!
     LogDebug("L1T") << "Unpacking AMC " << amc << " for BX " << bxid;

     auto res1_ = static_cast<CaloCollections*>(coll)->getMPJets();
     auto res2_ = static_cast<CaloCollections*>(coll)->getMPEtSums();
     auto res3_ = static_cast<CaloCollections*>(coll)->getMPEGammas();
     auto res4_ = static_cast<CaloCollections*>(coll)->getMPTaus();
     
     res1_->setBXRange(0,0);
     res2_->setBXRange(0,0);
     res3_->setBXRange(0,0);
     res4_->setBXRange(0,0);

     // Initialise frame indices for each data type
     int unsigned fet = 0;
     int unsigned fht = 1;
     int unsigned fjet = 6;
     int unsigned feg = 2;
     int unsigned ftau = 4;

     
     //      ===== Jets and Sums =====

     // ET / MET(x) / MET (y)

     uint32_t raw_data = block.payload()[fet];

     l1t::EtSum et = l1t::EtSum();

     et.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
     //et.setHwPt(raw_data & 0xFFFFF);
     switch(block.header().getID()){
     case 123:  et.setType(l1t::EtSum::kTotalEt);  break;
     case 121:  et.setType(l1t::EtSum::kTotalEtx); break;
     case 127:  et.setType(l1t::EtSum::kTotalEty); break;
     case 125:  et.setType(l1t::EtSum::kTotalEt);  break;
     case 131:  et.setType(l1t::EtSum::kTotalEtx); break;
     case 129: et.setType(l1t::EtSum::kTotalEty); break;
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
     case 123:  ht.setType(l1t::EtSum::kTotalHt);  break;
     case 121:  ht.setType(l1t::EtSum::kTotalHtx); break;
     case 127:  ht.setType(l1t::EtSum::kTotalHty); break;
     case 125:  ht.setType(l1t::EtSum::kTotalHt);  break;
     case 131:  ht.setType(l1t::EtSum::kTotalHtx); break;
     case 129: ht.setType(l1t::EtSum::kTotalHty); break;
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
       if ((block.header().getID() == 125) ||
           (block.header().getID() == 131) ||
           (block.header().getID() == 129)) {
         etasign = -1;
       }

       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;

       int mpEta = etasign*(raw_data & 0x3F);
       jet.setHwEta(CaloTools::caloEta(mpEta));
       jet.setHwPhi((raw_data >> 6) & 0x7F);
       jet.setHwPt((raw_data >> 13) & 0xFFFF);
         
       LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

       jet.setP4( l1t::CaloTools::p4MP(&jet) );
       res1_->push_back(0,jet);

       // Push them back in the right place (for checking sorting)
       /*
       int blockID = block.header().getID();
       int nPos=0, nNeg=0;
       for (unsigned i=0; i<res1_->size(0); i++)
         res1_->at(0,i).hwEta()>0 ? nPos++ : nNeg++;
       
       if (nJet==1) res1_->push_back(0,jet);
       else if (nJet==0) {
         if (blockID==123) {
           res1_->push_back(0,jet);
         }
         if (blockID==121) {
           if (nPos==1) res1_->push_back(0,jet);
           else if (nPos==2) res1_->insert(0,1,jet);
         }
         if (blockID==127) {
           if (nPos==2) res1_->push_back(0,jet);
           else if (nPos>2) res1_->insert(0,2,jet);
         }
         if (blockID==125) {
           res1_->push_back(0,jet);
         }
         if (blockID==131) {
           if (nNeg==1) res1_->push_back(0,jet);
           else if (nNeg==2) res1_->insert(0,nPos+1,jet);
         }
         if (blockID==129) {
           if (nNeg==2) res1_->push_back(0,jet);
           else if (nNeg>2) res1_->insert(0,nPos+2,jet);
         }
       }
       */
     }

     //      ===== EGammas =====
     
     // Two EGammas


     for (unsigned nEG=0; nEG < 2; nEG++){
       
       raw_data = block.payload()[feg+nEG];
       
       if (raw_data == 0)
	 continue;
       
       l1t::EGamma eg = l1t::EGamma();

       int etasign = 1;
       if ((block.header().getID() == 125) ||
           (block.header().getID() == 131) ||
           (block.header().getID() == 129)) {
         etasign = -1;
       }
       
       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;
       
       eg.setHwEta(etasign*((raw_data >> 4) & 0x3F));
       eg.setHwPhi((raw_data >> 10) & 0x7F);
       eg.setHwPt((raw_data >> 21) & 0x3FF);
       eg.setHwQual(((raw_data >> 3) & 0x1) + (((raw_data >> 1) & 0x1) << 2)); //ECalFG + EGammaLikeShape
       eg.setHwIso(raw_data & 0x1); 
	   
       LogDebug("L1T") << "Egamma: eta " << eg.hwEta() << " phi " << eg.hwPhi() << " pT " << eg.hwPt() << " qual " << eg.hwQual();
       
       eg.setP4( l1t::CaloTools::p4MP(&eg) );
       res3_->push_back(0,eg);
     }

     
     //      ===== Taus =====
     
     // Two taus

     for (unsigned nTau=0; nTau < 2; nTau++){
       
       raw_data = block.payload()[ftau+nTau];
       
       if (raw_data == 0)
	 continue;
       
       l1t::Tau tau = l1t::Tau();

       int etasign = 1;
       if ((block.header().getID() == 125) ||
           (block.header().getID() == 131) ||
           (block.header().getID() == 129)) {
         etasign = -1;
       }
       
       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;
       
       tau.setHwEta(etasign*((raw_data >> 4) & 0x3F));
       tau.setHwPhi((raw_data >> 10) & 0x7F);
       tau.setHwPt((raw_data >> 21) & 0x3FF);
       tau.setHwQual(((raw_data >> 3) & 0x1) + (((raw_data >> 1) & 0x1) << 2));
       tau.setHwIso(raw_data & 0x1);  
       

       // tau.setHwEta(etasign*((raw_data >> 9) & 0x7F));
       // tau.setHwPhi((raw_data >> 17) & 0xFF);
       // tau.setHwPt(raw_data & 0x1FF);
       // tau.setHwQual(((raw_data >> 26) & 0x1)); //ECalFG + TauLikeShape
       // tau.setHwIso(((raw_data >> 25) & 0x1) + ((raw_data >> 26) & 0x1) + ((raw_data >> 27) & 0x1)); 
	   
       LogDebug("L1T") << "Tau: eta " << tau.hwEta() << " phi " << tau.hwPhi() << " pT " << tau.hwPt() << " qual " << tau.hwQual();
       
       tau.setP4( l1t::CaloTools::p4MP(&tau) );
       res4_->push_back(0,tau);
     }



     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker);

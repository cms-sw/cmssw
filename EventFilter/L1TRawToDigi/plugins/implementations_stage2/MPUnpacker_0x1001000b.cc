#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloCollections.h"
#include "MPUnpacker_0x1001000b.h"

namespace l1t {
namespace stage2 {
  
   int MPUnpacker_0x1001000b::etaSign(int blkId){
     if ((blkId == BLK_TOT_NEG) ||
         (blkId == BLK_X_NEG) ||
	 (blkId == BLK_Y_NEG)) {
       return -1;
     }
     return 1;
   }

   bool
   MPUnpacker_0x1001000b::unpack(const Block& block, UnpackerCollections *coll)
   {

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize() << " AMC = " << block.amc().getAMCNumber();

     // check this is the correct MP
     unsigned int amc  = block.amc().getAMCNumber();
     unsigned int bxid = block.amc().getBX();
     //     if( (amc-1) != (bxid-1)%9 ) return true;
     //if( (amc-1) != ((bxid-1+3)%9) ) return true;   // temporary measure!
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
     int unsigned fet  = 0;
     int unsigned fht  = 2;
     int unsigned feg  = 3;
     int unsigned ftau = 5;
     int unsigned fjet = 7;



     //      ===== Jets and Sums =====

     // ET / MET(x) / MET (y)

     uint32_t raw_data = block.payload()[fet];

     l1t::EtSum et = l1t::EtSum();

     switch(block.header().getID()){
     case BLK_TOT_POS:
       et.setType(l1t::EtSum::kTotalEt);
       et.setHwPt( ( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
       break;
     case BLK_X_POS:
       et.setType(l1t::EtSum::kTotalEtx);
       et.setHwPt( ( uint32_t(raw_data) ) );
       break;
     case BLK_Y_POS:
       et.setType(l1t::EtSum::kTotalEty);
       et.setHwPt( ( uint32_t(raw_data) ) );
       break;
     case BLK_TOT_NEG:
       et.setType(l1t::EtSum::kTotalEt);
       et.setHwPt( ( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
       break;
     case BLK_X_NEG:
       et.setType(l1t::EtSum::kTotalEtx);
       et.setHwPt( ( uint32_t(raw_data) ) );
       break;
     case BLK_Y_NEG:
       et.setType(l1t::EtSum::kTotalEty); 
       et.setHwPt( ( uint32_t(raw_data) ) );
       break;
     default: 
       break;
     }

     LogDebug("L1T") << "ET/METx/METy: pT " << et.hwPt();

     res2_->push_back(0,et);


     // HT / MHT(x)/ MHT (y)

     raw_data = block.payload()[fht];

     l1t::EtSum ht = l1t::EtSum(); 

     //ht.setHwPt(raw_data & 0xFFFFF);
     ht.setHwPt( ( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
     switch(block.header().getID()){
     case BLK_TOT_POS:  ht.setType(l1t::EtSum::kTotalHt);  break;
     case BLK_X_POS:  ht.setType(l1t::EtSum::kTotalHtx); break;
     case BLK_Y_POS:  ht.setType(l1t::EtSum::kTotalHty); break;
     case BLK_TOT_NEG:  ht.setType(l1t::EtSum::kTotalHt);  break;
     case BLK_X_NEG:  ht.setType(l1t::EtSum::kTotalHtx); break;
     case BLK_Y_NEG: ht.setType(l1t::EtSum::kTotalHty); break;
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

       int etasign = etaSign(block.header().getID());

       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;

       int mpEta = etasign*(raw_data & 0x3F);
       jet.setHwEta(CaloTools::caloEta(mpEta));
       jet.setHwPhi((raw_data >> 6) & 0x7F);
       jet.setHwPt((raw_data >> 13) & 0xFFFF);
       jet.setHwQual((raw_data>>29) & 0x1 );
         
       LogDebug("L1T") << "Jet: eta " << jet.hwEta() << " phi " << jet.hwPhi() << " pT " << jet.hwPt() << " qual " << jet.hwQual();

       jet.setP4( l1t::CaloTools::p4MP(&jet) );
       res1_->push_back(0,jet);

     }

     //      ===== EGammas =====
     
     // Two EGammas


     for (unsigned nEG=0; nEG < 2; nEG++){
       
       raw_data = block.payload()[feg+nEG];
       
       if (raw_data == 0)
	 continue;
       
       l1t::EGamma eg = l1t::EGamma();

       int etasign = etaSign(block.header().getID());

       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;

       eg.setHwEta(etasign*((raw_data >> 3) & 0x3F));
       eg.setHwPhi((raw_data >> 9) & 0x7F);
       eg.setHwPt((raw_data >> 20) & 0xFFF);
       eg.setHwIso((raw_data>>1) & 0x3);       
       eg.setHwQual((raw_data>>16) & 0xf );

	   
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

       int etasign = etaSign(block.header().getID());
       
       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;
       
       tau.setHwEta(etasign*((raw_data >> 3) & 0x3F));
       tau.setHwPhi((raw_data >> 9) & 0x7F);
       tau.setHwPt((raw_data >> 20) & 0xFFF);
       tau.setHwIso((raw_data>>1) & 0x3);       
       tau.setHwQual((raw_data>>16) & 0xf );
       
       LogDebug("L1T") << "Tau: eta " << tau.hwEta() << " phi " << tau.hwPhi() << " pT " << tau.hwPt() << " qual " << tau.hwQual();
       
       tau.setP4( l1t::CaloTools::p4MP(&tau) );
       res4_->push_back(0,tau);
     }



     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker_0x1001000b);

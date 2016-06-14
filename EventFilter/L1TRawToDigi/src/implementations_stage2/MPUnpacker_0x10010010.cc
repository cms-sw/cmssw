#define EDM_ML_DEBUG 1

#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage2 {
      class MPUnpacker_0x10010010 : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   bool
   MPUnpacker_0x10010010::unpack(const Block& block, UnpackerCollections *coll)
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
     int unsigned fet  = 0;
     int unsigned fht  = 2;
     int unsigned feg  = 4;
     int unsigned ftau = 6;
     int unsigned fjet = 8;
     // int unsigned faux = 10;

     //      ===== Jets and Sums =====

     // ET / MET(x) / MET (y) with HF (configurable)

     uint32_t raw_data = block.payload()[fet];

     l1t::EtSum ethf = l1t::EtSum();

     switch(block.header().getID()){
     case 123: // 61
       ethf.setType(l1t::EtSum::kTotalEt2);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
       break;
     case 121: // 60
       ethf.setType(l1t::EtSum::kTotalEtx2);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 127: // 63
       ethf.setType(l1t::EtSum::kTotalEty2);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 125: // 62
       ethf.setType(l1t::EtSum::kTotalEt2);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
       break;
     case 131: // 65
       ethf.setType(l1t::EtSum::kTotalEtx2);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 129: // 64
       ethf.setType(l1t::EtSum::kTotalEty2); 
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     default: 
       break;
     }

     LogDebug("L1T") << "ET/METx/METy: pT " << ethf.hwPt();

     res2_->push_back(0,ethf);


     // ET / MET(x) / MET (y) without HF

     raw_data = block.payload()[fet + 1];

     l1t::EtSum etNoHF = l1t::EtSum();

     switch(block.header().getID()){
     case 123: // 61
       etNoHF.setType(l1t::EtSum::kTotalEt);
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
       break;
     case 121: // 60
       etNoHF.setType(l1t::EtSum::kTotalEtx);
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 127: // 63
       etNoHF.setType(l1t::EtSum::kTotalEty);
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 125: // 62
       etNoHF.setType(l1t::EtSum::kTotalEt);
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
       break;
     case 131: // 65
       etNoHF.setType(l1t::EtSum::kTotalEtx);
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 129: // 64
       etNoHF.setType(l1t::EtSum::kTotalEty); 
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     default:
       break;
     }

     LogDebug("L1T") << "ET/METx/METy (no HF): pT " << etNoHF.hwPt();

     res2_->push_back(0,etNoHF);


     // HT / MHT(x)/ MHT (y) with HF

     raw_data = block.payload()[fht];

     l1t::EtSum hthf = l1t::EtSum(); 

     //hthf.setHwPt(raw_data & 0xFFFFF);
     hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
     switch(block.header().getID()){
     case 123:  hthf.setType(l1t::EtSum::kTotalHt2);  break;
     case 121:  hthf.setType(l1t::EtSum::kTotalHtx2); break;
     case 127:  hthf.setType(l1t::EtSum::kTotalHty2); break;
     case 125:  hthf.setType(l1t::EtSum::kTotalHt2);  break;
     case 131:  hthf.setType(l1t::EtSum::kTotalHtx2); break;
     case 129:  hthf.setType(l1t::EtSum::kTotalHty2); break;
     default: break;
     }

     LogDebug("L1T") << "HTHF/MHTHFx/MHTHFy: pT " << hthf.hwPt();

     res2_->push_back(0,hthf);


     // HT / MHT(x)/ MHT (y) no HF

     raw_data = block.payload()[fht+1];

     l1t::EtSum htNoHF = l1t::EtSum(); 

     //htNoHF.setHwPt(raw_data & 0xFFFFF);
     htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFFFF) << 16 ) >> 16 );
     switch(block.header().getID()){
     case 123:  htNoHF.setType(l1t::EtSum::kTotalHt);  break;
     case 121:  htNoHF.setType(l1t::EtSum::kTotalHtx); break;
     case 127:  htNoHF.setType(l1t::EtSum::kTotalHty); break;
     case 125:  htNoHF.setType(l1t::EtSum::kTotalHt);  break;
     case 131:  htNoHF.setType(l1t::EtSum::kTotalHtx); break;
     case 129:  htNoHF.setType(l1t::EtSum::kTotalHty); break;
     default: break;
     }

     LogDebug("L1T") << "HTNOHF/MHTNOHFx/MHTNOHFy: pT " << htNoHF.hwPt();

     res2_->push_back(0,htNoHF);


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
       jet.setHwQual((raw_data>>29) & 0x1 );

       if (jet.hwPt()==0) continue;
         
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

       int etasign = 1;
       if ((block.header().getID() == 125) ||
           (block.header().getID() == 131) ||
           (block.header().getID() == 129)) {
         etasign = -1;
       }
       
       LogDebug("L1") << "block ID=" << block.header().getID() << " etasign=" << etasign;

       eg.setHwEta(etasign*((raw_data >> 3) & 0x3F));
       eg.setHwPhi((raw_data >> 9) & 0x7F);
       eg.setHwPt((raw_data >> 20) & 0xFFF);
       eg.setHwIso((raw_data>>1) & 0x3);       
       eg.setHwQual((raw_data>>16) & 0xf );

       if (eg.hwPt()==0) continue;
	   
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
       
       tau.setHwEta(etasign*((raw_data >> 3) & 0x3F));
       tau.setHwPhi((raw_data >> 9) & 0x7F);
       tau.setHwPt((raw_data >> 20) & 0xFFF);
       tau.setHwIso((raw_data>>1) & 0x3);       
       tau.setHwQual((raw_data>>16) & 0xf );

       if (tau.hwPt()==0) continue;
       
       LogDebug("L1T") << "Tau: eta " << tau.hwEta() << " phi " << tau.hwPhi() << " pT " << tau.hwPt() << " qual " << tau.hwQual();
       
       tau.setP4( l1t::CaloTools::p4MP(&tau) );
       res4_->push_back(0,tau);
     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker_0x10010010);

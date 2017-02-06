#define EDM_ML_DEBUG 1

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"

#include "CaloCollections.h"
#include "L1TStage2Layer2Constants.h"

#include "MPUnpacker_0x10010010.h"

namespace l1t {
namespace stage2 {
   bool
   MPUnpacker_0x10010010::unpack(const Block& block, UnpackerCollections *coll)
   {

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize() << " AMC = " << block.amc().getAMCNumber();

     // check this is the correct MP
     const unsigned int tmt  = block.amc().getBoardID() - l1t::stage2::layer2::mp::offsetBoardId + 1;
     const unsigned int bxid = block.amc().getBX();

     // handle offset between BC0 marker and actual BC0...
     if( (tmt-1) != ((bxid-1+3)%9) ) return true;
     LogDebug("L1T") << "Unpacking TMT # " << tmt << " for BX " << bxid;

     auto res1_ = static_cast<CaloCollections*>(coll)->getMPJets();
     auto res2_ = static_cast<CaloCollections*>(coll)->getMPEtSums();
     auto res3_ = static_cast<CaloCollections*>(coll)->getMPEGammas();
     auto res4_ = static_cast<CaloCollections*>(coll)->getMPTaus();
     
     res1_->setBXRange(0,0);
     res2_->setBXRange(0,0);
     res3_->setBXRange(0,0);
     res4_->setBXRange(0,0);

     // Initialise frame indices for each data type
     const int unsigned fet  = 0;
     const int unsigned fht  = 2;
     const int unsigned feg  = 4;
     const int unsigned ftau = 6;
     const int unsigned fjet = 8;
     const int unsigned faux = 10;

     //      ===== Jets and Sums =====

     // ET / MET(x) / MET (y) with HF (configurable)

     uint32_t raw_data = block.payload()[fet];

     l1t::EtSum ethf = l1t::EtSum();

     switch(block.header().getID()){
     case 123: // 61
       ethf.setType(l1t::EtSum::kTotalEtHF);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
       break;
     case 121: // 60
       ethf.setType(l1t::EtSum::kTotalEtxHF);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 127: // 63
       ethf.setType(l1t::EtSum::kTotalEtyHF);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 125: // 62
       ethf.setType(l1t::EtSum::kTotalEtHF);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
       break;
     case 131: // 65
       ethf.setType(l1t::EtSum::kTotalEtxHF);
       ethf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 129: // 64
       ethf.setType(l1t::EtSum::kTotalEtyHF); 
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
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
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
       etNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
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

     switch(block.header().getID()){
     case 123: // 61
       hthf.setType(l1t::EtSum::kTotalHtHF);
       hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
       break;
     case 121: // 60
       hthf.setType(l1t::EtSum::kTotalHtxHF);
       hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 127: // 63
       hthf.setType(l1t::EtSum::kTotalHtyHF);
       hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 125: // 62
       hthf.setType(l1t::EtSum::kTotalHtHF);
       hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
       break;
     case 131: // 65
       hthf.setType(l1t::EtSum::kTotalHtxHF);
       hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 129: // 64
       hthf.setType(l1t::EtSum::kTotalHtyHF); 
       hthf.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     default: 
       break;
     }

     LogDebug("L1T") << "HTHF/MHTHFx/MHTHFy: pT " << hthf.hwPt();

     res2_->push_back(0,hthf);


     // HT / MHT(x)/ MHT (y) no HF

     raw_data = block.payload()[fht+1];

     l1t::EtSum htNoHF = l1t::EtSum(); 

     switch(block.header().getID()){
     case 123: // 61
       htNoHF.setType(l1t::EtSum::kTotalHt);
       htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
       break;
     case 121: // 60
       htNoHF.setType(l1t::EtSum::kTotalHtx);
       htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 127: // 63
       htNoHF.setType(l1t::EtSum::kTotalHty);
       htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 125: // 62
       htNoHF.setType(l1t::EtSum::kTotalHt);
       htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data & 0xFFFF)) );
       break;
     case 131: // 65
       htNoHF.setType(l1t::EtSum::kTotalHtx);
       htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     case 129: // 64
       htNoHF.setType(l1t::EtSum::kTotalHty); 
       htNoHF.setHwPt( static_cast<int32_t>( uint32_t(raw_data) ) );
       break;
     default:
       break;
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

      //      ===== Aux =====
      raw_data = block.payload()[faux];

      // create a sum object for each type of HF sum
      l1t::EtSum mbp0 = l1t::EtSum();
      l1t::EtSum mbm0 = l1t::EtSum();
      l1t::EtSum mbm1 = l1t::EtSum();
      l1t::EtSum mbp1 = l1t::EtSum();

      // readout the sums only if the correct block is  being processed (first frame of AUX)
      switch(block.header().getID()){
      case 121: // this should correspond to the first link
        // read 4 bits starting at position 24 (24 -> 28)
        mbp0.setHwPt( ( raw_data >> 24 ) & 0xF );
        mbp0.setType( l1t::EtSum::kMinBiasHFP0 );

        // read 4 bits starting at position 16 (16 -> 20)
        mbm0.setHwPt( ( raw_data >> 16 ) & 0xF );
        mbm0.setType( l1t::EtSum::kMinBiasHFM0 );

        // read 4 bits starting at position 8 (8 -> 12)
        mbp1.setHwPt( ( raw_data >> 8 ) & 0xF );
        mbp1.setType( l1t::EtSum::kMinBiasHFP1 );

        // read the first 4 bits by masking with 0xF
        mbm1.setHwPt( raw_data & 0xF );
        mbm1.setType( l1t::EtSum::kMinBiasHFM1 );

        LogDebug("L1T") << "mbp0 HF sum: " << mbp0.hwPt();
        LogDebug("L1T") << "mbm0 HF sum: " << mbm0.hwPt();
        LogDebug("L1T") << "mbp1 HF sum: " << mbp1.hwPt();
        LogDebug("L1T") << "mbm1 HF sum: " << mbm1.hwPt();

        res2_->push_back(0,mbp0);
        res2_->push_back(0,mbm0);
        res2_->push_back(0,mbp1);
        res2_->push_back(0,mbm1);
        break;
      default:
        break;
      }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::MPUnpacker_0x10010010);

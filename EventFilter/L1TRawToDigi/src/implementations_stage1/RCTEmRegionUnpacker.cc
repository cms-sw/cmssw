#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "PhysicsToBitConverter.h"
#include "rctDataBase.h"


#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"


#include <iostream>
#include <fstream>

#include "CaloCollections.h"

namespace l1t {
  namespace stage1 {
    class RCTEmRegionUnpacker : public Unpacker {
      public:
        virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      private:
        unsigned int counter_ = 0;
    };
  }
}

// Implementation

namespace l1t {
   namespace stage1 {
      void
      unpack_em(const Block& block, UnpackerCollections *coll)
      {
         int nBX, firstBX, lastBX;
         nBX = int(ceil(block.header().getSize() / 6.)); 
         getBXRange(nBX, firstBX, lastBX);

         auto resRCTEMCands_ = static_cast<CaloCollections*>(coll)->getCaloEmCands();

         // Initialise index
         int unsigned i = 0;

         for (int bx=firstBX; bx<=lastBX; bx++){

            unsigned int crate;
            bool even=0;

            std::vector <uint32_t> uint;
            uint.reserve(6);

            PhysicsToBitConverter converter;
            rctDataBase database;
            int mp7link=(int)(block.header().getID()/2);
            database.GetLinkRCT(mp7link,crate,even);

            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);

            LogDebug("L1T")<<"--------------- mp7 link ="<<mp7link<<"RCT crate id="<<crate<<", RCT crate even="<<even<<std::endl;

            if(!even) {
               for(int i=0;i<6;i++) converter.Set32bitWordLinkOdd(i,uint[i]);
               converter.Convert();

               for(int j = 0; j < 4; j++) {

                  unsigned int rank=(unsigned int)converter.GetNEEt(j);
                  unsigned int reg=(unsigned int)converter.GetNEReg(j);
                  unsigned int card=(unsigned int)converter.GetNECard(j);

                  LogDebug("L1T")<<"UNPACKER, CRATE"<<crate<<"NON ISO em rank="<<rank<<", region="<<reg<<", card="<<card<<std::endl;

                  L1CaloEmCand em = L1CaloEmCand(rank,reg,card,crate,false,j,bx);
                  resRCTEMCands_->push_back(em);
                  }

               for(int j = 0; j < 4; j++) {

                  unsigned int rank=converter.GetIEEt(j);
                  unsigned int reg=converter.GetIEReg(j);
                  unsigned int card=converter.GetIECard(j);

                  LogDebug("L1T")<<"UNPACKER, CRATE"<<crate<<"ISO em rank="<<rank<<", region="<<reg<<", card="<<card<<std::endl;
                  L1CaloEmCand em = L1CaloEmCand(rank,reg,card,crate,true,j,bx);
                  resRCTEMCands_->push_back(em);
               }
            }// end if odd
         }// end of loop over BX
      }

      void
      unpack_region(const Block& block, UnpackerCollections *coll)
      {
         int nBX, firstBX, lastBX;
         nBX = int(ceil(block.header().getSize() / 6.)); 
         getBXRange(nBX, firstBX, lastBX);

         auto resRCTRegions_ = static_cast<CaloCollections*>(coll)->getCaloRegions();

         // Initialise index
         int unsigned i = 0;
         std::vector <uint32_t> uint;
         uint.reserve(6);

         for (int bx=firstBX; bx<=lastBX; bx++){

            unsigned int crate;
            bool even=0;

            PhysicsToBitConverter converter;
            rctDataBase database;
            int mp7link=(int)(block.header().getID()/2);
            database.GetLinkRCT(mp7link,crate,even);

            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);
            uint.push_back(block.payload()[i++]);

            LogDebug("L1T")<<"--------------- mp7 link ="<<mp7link<<"RCT crate id="<<crate<<", RCT crate even="<<even<<std::endl;

            if(!even) {

               for(int i=0;i<6;i++) converter.Set32bitWordLinkOdd(i,uint[i]);
               converter.Convert();

               for(int j = 0; j < 8; j++) {

                  unsigned int hfet=(unsigned int)converter.GetHFEt(j);

                  LogDebug("L1T")<<"UNPACKER, CRATE"<<crate<<"region="<<j<<", rgnEt="<<hfet<<std::endl;
                  L1CaloRegion rgn = L1CaloRegion(hfet,0,crate,j);
                  rgn.setBx(bx);
                  resRCTRegions_->push_back(rgn);
               }
            }// end if odd


            else{

               for(int i=0;i<6;i++) converter.Set32bitWordLinkEven(i,uint[i]);
               converter.Convert();

               for(int j = 0; j < 7; j++) {
                  for(int k = 0; k < 2; k++) {

                     unsigned int RCet=(unsigned int)converter.GetRCEt(j,k);
                     bool overflow=(bool)converter.GetRCOf(j,k);
                     bool tauveto=(bool)converter.GetRCTau(j,k);
                     bool hadveto=(bool)converter.GetRCHad(j,k);
                     bool quiet=false;                                //to be fixed

                     LogDebug("L1T")<<"UNPACKER, CRATE="<<crate<<",region="<<k<<", card="<<j<<", rgnEt="<<RCet<<", overflow="<<overflow<<", tauveto="<<tauveto<<", hadveto="<<hadveto<<std::endl;
                     L1CaloRegion rgn = L1CaloRegion(RCet,overflow,tauveto,hadveto,quiet,crate,j,k);    
                     rgn.setBx(bx); 
                     resRCTRegions_->push_back(rgn);
                  }
               }
            }// end if even
         }// end of loop over BX 
      }

      bool RCTEmRegionUnpacker::unpack(const Block& block, UnpackerCollections *coll){
         if (block.header().getCapID() == 0) {
            unpack_region(block, coll);
         } else if (block.header().getCapID() == 1) {
            unpack_em(block, coll);
         } else {
            return false;
         }
         return true;
      }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::RCTEmRegionUnpacker);

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
    class RCTEmUnpacker : public Unpacker {
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
    bool RCTEmUnpacker::unpack(const Block& block, UnpackerCollections *coll){

      int nBX = int(ceil(block.header().getSize() / 6.)); 

      // Find the first and last BXs

      int firstBX = -(ceil((double)nBX/2.)-1);
      int lastBX;
      if (nBX % 2 == 0) {
        lastBX = ceil((double)nBX/2.)+1;
      } else {
        lastBX = ceil((double)nBX/2.);
      }

      auto resRCTEMCands_ = static_cast<CaloCollections*>(coll)->getCaloEmCands();
      resRCTEMCands_->resize(144*nBX);


      // Initialise index
      int unsigned i = 0;

      for (int bx=firstBX; bx<lastBX; bx++){

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
            resRCTEMCands_->erase(resRCTEMCands_->begin()+crate*8+2*j+1);
            resRCTEMCands_->insert(resRCTEMCands_->begin()+crate*8+2*j+1,em);
          }

          for(int j = 0; j < 4; j++) {
            
            unsigned int rank=converter.GetIEEt(j);
            unsigned int reg=converter.GetIEReg(j);
            unsigned int card=converter.GetIECard(j);

            LogDebug("L1T")<<"UNPACKER, CRATE"<<crate<<"ISO em rank="<<rank<<", region="<<reg<<", card="<<card<<std::endl;
            L1CaloEmCand em = L1CaloEmCand(rank,reg,card,crate,true,j,bx);
            resRCTEMCands_->erase(resRCTEMCands_->begin()+crate*8+2*j);
            resRCTEMCands_->insert(resRCTEMCands_->begin()+crate*8+2*j,em); 
          }
        }// end if odd
      }// end of loop over BX
      return true;
    }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::RCTEmUnpacker);

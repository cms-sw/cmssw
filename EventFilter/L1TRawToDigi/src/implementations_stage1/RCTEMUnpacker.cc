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
      resRCTEMCands_->setBXRange(std::min(firstBX, resRCTEMCands_->getFirstBX()), std::max(lastBX, resRCTEMCands_->getLastBX()));

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

          for(int j = 0; j < 4; j++) {

            for(int i=0;i<6;i++) converter.Set32bitWordLinkOdd(i,uint[i]);
            converter.Convert();
            
            unsigned int rank=(unsigned int)converter.GetNEEt(j);
            unsigned int reg=(unsigned int)converter.GetNEReg(j);
            unsigned int card=(unsigned int)converter.GetNECard(j);

            LogDebug("L1T") <<"index="<<j<<", neRank="<<rank<<", neRegn="<<reg<<", neCard="<<card<<std::endl;
            L1CaloEmCand em = L1CaloEmCand(rank,reg,card,crate,false);
            ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 =new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
            CaloEmCand EmCand(*p4,(int) em.rank(),(int) em.regionId().ieta(),(int) em.regionId().iphi(),(int) j);            //j was originally em.index, to be checked 
            EmCand.setHwIso((int) em.isolated());
            resRCTEMCands_->push_back(bx,EmCand);
          }

          for(int j = 0; j < 4; j++) {

            for(int i=0;i<6;i++) converter.Set32bitWordLinkEven(i,uint[i]);
            converter.Convert();
            
            unsigned int rank=converter.GetIEEt(j);
            unsigned int reg=converter.GetIEReg(j);
            unsigned int card=converter.GetIECard(j);

            LogDebug("L1T") <<"index="<<j<<", neRank="<<rank<<", neRegn="<<reg<<", neCard="<<card<<std::endl;
            L1CaloEmCand em = L1CaloEmCand(rank,reg,card,crate,true);
            ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > *p4 =new ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >();
            CaloEmCand EmCand(*p4,(int) em.rank(),(int) em.regionId().ieta(),(int) em.regionId().iphi(),(int) j);            //j was originally em.index, to be checked 
            EmCand.setHwIso((int) em.isolated());
            resRCTEMCands_->push_back(bx,EmCand);
          }
        }// end if odd
      }// end of loop over BX
      return true;
    }
  }
}

DEFINE_L1T_UNPACKER(l1t::stage1::RCTEmUnpacker);

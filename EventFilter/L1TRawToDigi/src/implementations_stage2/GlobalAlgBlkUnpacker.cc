#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "GTCollections.h"

namespace l1t {
   namespace stage2 {
      class GlobalAlgBlkUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   bool
   GlobalAlgBlkUnpacker::unpack(const Block& block, UnpackerCollections *coll)
   {

     LogDebug("L1T") << "Block ID  = " << block.header().getID() << " size = " << block.header().getSize();

     int nBX = int(ceil(block.header().getSize() / 6.)); // FOR GT Not sure what we have here...put at 6 because of 6 frames   Since there are 12 EGamma objects reported per event (see CMS IN-2013/005)

     // Find the central, first and last BXs
     int firstBX = -(ceil((double)nBX/2.)-1);
     int lastBX;
     if (nBX % 2 == 0) {
       lastBX = ceil((double)nBX/2.);
     } else {
       lastBX = ceil((double)nBX/2.)-1;
     }

     auto res_ = static_cast<GTCollections*>(coll)->getAlgs();
     res_->setBXRange(firstBX, lastBX);

     LogDebug("L1T") << "nBX = " << nBX << " first BX = " << firstBX << " lastBX = " << lastBX;

     // Loop over multiple BX and then number of EG cands filling collection
     for (int bx=firstBX; bx<=lastBX; bx++){

        // If this is the first block, instantiate GlobalAlg so it is there to fill from mult. blocks
       if(block.header().getID()==1) {

	  LogDebug("L1T") << "Creating GT Algorithm Block for BX =" << bx;
          GlobalAlgBlk talg = GlobalAlgBlk();
          res_->push_back(bx,talg);

       }

       //fetch
       GlobalAlgBlk alg = res_->at(bx,0);

       //Determine offset of algorithm bits based on block.ID
       // ID=0 offset = 0;  ID=3 offset=192;  ID=5 offset=384=2*192;
       int algOffset = (block.header().getID()/2)*192;

       for(unsigned int wd=0;  wd<block.header().getSize(); wd++) {
         uint32_t raw_data = block.payload()[wd];
	 LogDebug("L1T") << " payload word " << wd << " 0x" << hex << raw_data << " offset=" << algOffset;

         //parse these 32 bits into algorithm bits (perhaps needs a more efficient way of doing this?
         if(block.header().getID()!=5 || wd<4) {
           for(unsigned int bt=0; bt<32; bt++) {
	     int val = ((raw_data >> bt) & 0x1);
             if(val==1) alg.setAlgoDecisionFinal(bt+wd*32+algOffset,true);
           }
	 } else if(block.header().getID()==5 && wd==4) {
           //This is the FINOR
           alg.setFinalOR(raw_data);
           LogDebug("L1T") << " Packing the FinalOR " << wd << " 0x" << hex << raw_data;	
         }
       }

       // Put the object back into place (Must be better way)
       res_->set(bx,0,alg);

       //alg.print(std::cout);

     }

     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage2::GlobalAlgBlkUnpacker);

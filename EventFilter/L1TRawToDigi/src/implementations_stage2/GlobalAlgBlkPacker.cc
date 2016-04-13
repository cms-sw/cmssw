#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "GTTokens.h"

namespace l1t {
   namespace stage2 {
      class GlobalAlgBlkPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   GlobalAlgBlkPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<GlobalAlgBlkBxCollection> algs;
      event.getByToken(static_cast<const GTTokens*>(toks)->getAlgToken(), algs);

      unsigned int wdPerBX = 6; //should this be configured someplace else?

      Blocks res;

      for(int blk=0; blk<9; blk++) {
        
	unsigned int blkID = blk*2+33; 
	
	unsigned int algOffset = (2*blk+1)/2;
	algOffset = (algOffset%3)*192;

        //vector of words
        std::vector<uint32_t> load;
      
        for (int i = algs->getFirstBX(); i <= algs->getLastBX(); ++i) {
        
          for (auto j = algs->begin(i); j != algs->end(i); ++j) {
	  	   	     
              for(unsigned int wd=0; wd<wdPerBX; wd++) {

		uint32_t word = 0;
		
                if( (blk+1)%3 > 0 || wd<4 ) {

                  unsigned int startAlg = wd*32+algOffset;
		  for(unsigned bt=0; bt<32; bt++) {
		  
		     if(blk<3) {
		       if(j->getAlgoDecisionInitial(bt+startAlg))    word |= (0x1 << bt);
		     } else if(blk<6) {
		       if(j->getAlgoDecisionInterm(bt+startAlg))  word |= (0x1 << bt);
		     } else {
		       if(j->getAlgoDecisionFinal(bt+startAlg))      word |= (0x1 << bt);
		     }    
		  }
		
		} else if(blk==2 && (wd==4 || wd==5) ) {  
		 
		   //putting hashed values of the menu name and firmware uuid into record.
		   if(wd==4) word |= (j->getL1MenuUUID() & 0xFFFFFFFF);
		   if(wd==5) word |= (j->getL1FirmwareUUID() & 0xFFFFFFFF);
		   
		} else if(blk==8){
		
		  if(wd == 4) {
		    if(j->getFinalOR())         word |= (0x1 << 16);
		    if(j->getFinalORVeto())     word |= (0x1 << 8);
		    if(j->getFinalORPreVeto())  word |= (0x1 << 0);
		  } else if (wd == 5) {
		    word |= (j->getPreScColumn() & 0xFF);
		  }		
		} //if on blk  
		
			   
	        load.push_back(word);
	      } //loop over words

           } //end loop over alg objects.(trivial one per BX)

        } //end loop over bx
      
        res.push_back(Block(blkID, load));
      
      } //loop over blks

      return res;
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::GlobalAlgBlkPacker);

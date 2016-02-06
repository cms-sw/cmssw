#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "GTTokens.h"

namespace l1t {
   namespace stage2 {
      class GlobalExtBlkPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   GlobalExtBlkPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<GlobalExtBlkBxCollection> exts;
      event.getByToken(static_cast<const GTTokens*>(toks)->getExtToken(), exts);

      unsigned int wdPerBX = 6; //should this be configured someplace else?

      Blocks res;

      for(int blk=0; blk<4; blk++) {
        
	unsigned int blkID = blk*2+24; 
	
	unsigned int extOffset = blk*64;

        //vector of words
        std::vector<uint32_t> load;
      
        for (int i = exts->getFirstBX(); i <= exts->getLastBX(); ++i) {
        
          for (auto j = exts->begin(i); j != exts->end(i); ++j) {
	  	   	     
              for(unsigned int wd=0; wd<wdPerBX; wd++) {

		uint32_t word = 0;
		
                if( wd<2 ) {

                  unsigned int startExt = wd*32+extOffset;
		  for(unsigned bt=0; bt<32; bt++) {
		  
		       if(j->getExternalDecision(bt+startExt))    word |= (0x1 << bt);

		  } //end loop over bits
		} //endif wrd < 2   
			   
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

DEFINE_L1T_PACKER(l1t::stage2::GlobalExtBlkPacker);

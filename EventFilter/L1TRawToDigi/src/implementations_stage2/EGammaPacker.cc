#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

#include "L1TStage2Layer2Constants.h"

namespace l1t {
   namespace stage2 {
      class EGammaPacker : public Packer {
         public:
	    EGammaPacker(int b1, int b2) : b1_(b1), b2_(b2) {}
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
	    int b1_, b2_;
      };

      class GTEGammaPacker : public EGammaPacker {
         public:
             GTEGammaPacker() : EGammaPacker(8,10) {}
      };
      class CaloEGammaPacker : public EGammaPacker {
         public:
	     CaloEGammaPacker() : EGammaPacker(9,11) {}
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   EGammaPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<EGammaBxCollection> egs;
      event.getByToken(static_cast<const CommonTokens*>(toks)->getEGammaToken(), egs);

      std::vector<uint32_t> load1, load2;

      for (int i = egs->getFirstBX(); i <= egs->getLastBX(); ++i) {
	
	for (auto j = egs->begin(i); j != egs->end(i); ++j) {

	  uint32_t packed_eta = abs(j->hwEta()) & 0x7F;
	  if (j->hwEta() < 0){
	    packed_eta = (128 - packed_eta) | 1<<7;
	  }

	  uint32_t word =					\
	    std::min(j->hwPt(), 0x1FF) |
	    packed_eta << 9 |
	    (j->hwPhi() & 0xFF) << 17 |
	    (j->hwIso() & 0x3) << 25 |
	    (j->hwQual() & 0x7) << 27;
	
	  if (load1.size() < l1t::stage2::layer2::demux::nEGPerLink) load1.push_back(word);
	  else load2.push_back(word);
	
	}
      }
      
      // push zeroes if jets are missing                                      
      while (load1.size()<l1t::stage2::layer2::demux::nOutputFramePerBX) load1.push_back(0);
      
      while (load2.size()<l1t::stage2::layer2::demux::nOutputFramePerBX) load2.push_back(0);
  
      return {Block(b1_, load1), Block(b2_, load2)};

   }

}
}

DEFINE_L1T_PACKER(l1t::stage2::GTEGammaPacker);
DEFINE_L1T_PACKER(l1t::stage2::CaloEGammaPacker);

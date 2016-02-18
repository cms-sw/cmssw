#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "CaloTokens.h"

#include "L1TStage2Layer2Constants.h"

namespace l1t {
   namespace stage2 {
      class JetPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage2 {
   Blocks
   JetPacker::pack(const edm::Event& event, const PackerTokens* toks)
   {
      edm::Handle<JetBxCollection> jets;
      event.getByToken(static_cast<const CaloTokens*>(toks)->getJetToken(), jets);

      std::vector<uint32_t> load1, load2;

      // loop over BX
      for (int i = jets->getFirstBX(); i <= jets->getLastBX(); ++i) {

	 // get jets from this BX
	for (auto j = jets->begin(i); j != jets->end(i); ++j ) {

	  uint32_t packed_eta = abs(j->hwEta()) & 0x7F;
	  if (j->hwEta() < 0){
	    packed_eta = (128 - packed_eta) | 1<<7;
	  }

	  uint32_t word =			\
	    std::min(j->hwPt(), 0x7FF) |
	    packed_eta << 11 |
	    (j->hwPhi() & 0xFF) << 19 |
	    (j->hwQual() & 0x7) << 27;
	  
	  if (load1.size() < l1t::stage2::layer2::demux::nJetPerLink) load1.push_back(word);
	  else load2.push_back(word);
		 
	}
	 

	// push zeroes if jets are missing
	while (load1.size()<l1t::stage2::layer2::demux::nOutputFramePerBX) load1.push_back(0);
	
	while (load2.size()<l1t::stage2::layer2::demux::nOutputFramePerBX) load2.push_back(0);

      }

      

      return {Block(13, load1), Block(15, load2)};
   }
}
}

DEFINE_L1T_PACKER(l1t::stage2::JetPacker);

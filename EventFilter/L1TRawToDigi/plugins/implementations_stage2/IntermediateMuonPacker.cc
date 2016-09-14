#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

#include "L1Trigger/L1TMuon/interface/MuonRawDigiTranslator.h"
#include "GMTTokens.h"

namespace l1t {
   namespace stage2 {
      class IntermediateMuonPacker : public Packer {
         public:
            virtual Blocks pack(const edm::Event&, const PackerTokens*) override;
      };
   }
}

// Implementation
namespace l1t {
   namespace stage2 {
      Blocks
      IntermediateMuonPacker::pack(const edm::Event& event, const PackerTokens* toks)
      {	 

	 //auto imdBmtfToken = static_cast<const GMTTokens*>(toks)->getImdMuonTokenBMTF();
         //auto imdOmtfNegToken = static_cast<const GMTTokens*>(toks)->getImdMuonTokenOMTFNeg();
         //auto imdOmtfPosToken = static_cast<const GMTTokens*>(toks)->getImdMuonTokenOMTFPos();
	 //auto imdEmtfNegToken = static_cast<const GMTTokens*>(toks)->getImdMuonTokenEMTFNeg();
	 //auto imdEmtfPosToken = static_cast<const GMTTokens*>(toks)->getImdMuonTokenEMTFPos();

         Blocks blocks;

         return blocks;
      }
   }
}

DEFINE_L1T_PACKER(l1t::stage2::IntermediateMuonPacker);

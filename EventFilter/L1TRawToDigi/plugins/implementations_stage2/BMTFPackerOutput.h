#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include "BMTFTokens.h"

namespace l1t {
   namespace stage2 {
      class BMTFPackerOutput : public Packer 
      {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
         private:
            std::map<unsigned int, std::vector<uint32_t> > payloadMap_;
            
      };
   }
}

#include "FWCore/Framework/interface/Event.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h" 

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"

#include "BMTFTokens.h"

namespace l1t {
   namespace stage2 {
      class BMTFPackerInputs : public Packer 
      {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
         private:
            std::map<unsigned int, std::vector<uint32_t> > payloadMap_;

            uint32_t wordPhMaker(const L1MuDTChambPhDigi& phInput);
            uint32_t wordThMaker(const L1MuDTChambThDigi& thInput, const bool& qualFlag);

            static const unsigned int phiMask = 0xFFF;
            static const unsigned int phiShift = 0;
            static const unsigned int phiBMask = 0x3FF;
            static const unsigned int phiBShift = 12;
            static const unsigned int qualMask = 0x7;
            static const unsigned int qualShift = 22;
            static const unsigned int rpcMask = 0x1;
            static const unsigned int rpcShift = 26;
            static const unsigned int bxCntMask = 0x3;
            static const unsigned int bxCntShift = 30;

            static const int ownLinks_[];

      };
   }
}

#ifndef L1T_PACKER_STAGE2_MUONPACKER_H
#define L1T_PACKER_STAGE2_MUONPACKER_H

#include <map>
#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class MuonPacker : public Packer {
         public:
	    MuonPacker(unsigned b1) : b1_(b1) {}
            Blocks pack(const edm::Event&, const PackerTokens*) override;
            unsigned b1_;
         private:
            typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;
      };

      class GTMuonPacker : public MuonPacker {
         public:
             GTMuonPacker() : MuonPacker(0) {}
      };
      class GMTMuonPacker : public MuonPacker {
         public:
	     GMTMuonPacker() : MuonPacker(1) {}
      };
   }
}

#endif

#ifndef L1T_PACKER_STAGE2_TAUPACKER_H
#define L1T_PACKER_STAGE2_TAUPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class TauPacker : public Packer {
         public:
	    TauPacker(int b1, int b2) : b1_(b1), b2_(b2) {}
            Blocks pack(const edm::Event&, const PackerTokens*) override;
	    int b1_, b2_;
      };

      class GTTauPacker : public TauPacker {
         public:
             GTTauPacker() : TauPacker(16,18) {}
      };
      class CaloTauPacker : public TauPacker {
         public:
	     CaloTauPacker() : TauPacker(17,19) {}
      };

   }
}

#endif

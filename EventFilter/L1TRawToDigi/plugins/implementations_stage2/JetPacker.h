#ifndef L1T_PACKER_STAGE2_JETPACKER_H
#define L1T_PACKER_STAGE2_JETPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Packer.h"

namespace l1t {
   namespace stage2 {
      class JetPacker : public Packer {
         public:
	    JetPacker(int b1, int b2) : b1_(b1), b2_(b2) {}
            Blocks pack(const edm::Event&, const PackerTokens*) override;
	    int b1_, b2_;
      };

      class GTJetPacker : public JetPacker {
         public:
             GTJetPacker() : JetPacker(12,14) {}
      };
      class CaloJetPacker : public JetPacker {
         public:
	     CaloJetPacker() : JetPacker(13,15) {}
      };


   }
}

#endif

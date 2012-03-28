#ifndef MuonReco_MuonRPCHitMatch_h
#define MuonReco_MuonRPCHitMatch_h

#include <cmath>

namespace reco {
   class MuonRPCHitMatch {
      public:
         float x;              // X position of the matched segment
         unsigned int mask;    // arbitration mask
         int bx;               // bunch crossing

	 MuonRPCHitMatch():x(0),mask(0),bx(0){}
   };
}

#endif

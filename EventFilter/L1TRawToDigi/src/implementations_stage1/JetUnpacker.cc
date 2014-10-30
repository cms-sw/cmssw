#include "FWCore/Framework/interface/MakerMacros.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "CaloCollections.h"

namespace l1t {
   namespace stage1 {
      class JetUnpacker : public Unpacker {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
      };
   }
}

// Implementation

namespace l1t {
namespace stage1 {
   bool
   JetUnpacker::unpack(const Block& block, UnpackerCollections *coll)
   {
     return true;
   }
}
}

DEFINE_L1T_UNPACKER(l1t::stage1::JetUnpacker);

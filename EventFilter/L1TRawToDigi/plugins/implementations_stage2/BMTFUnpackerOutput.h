#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "BMTFCollections.h"

namespace l1t{
  namespace stage2{
    class BMTFUnpackerOutput : public Unpacker
    {
    public:
      bool unpack(const Block& block, UnpackerCollections *coll) override;
    };
  }
}

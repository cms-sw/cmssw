#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t{
   namespace stage2{
      struct qualityHits
      {
         int linkNo;
         int hits[3][7];
      };

      class BMTFUnpackerInputsOldQual : public Unpacker
      {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
         private:
            std::map<int, qualityHits> linkAndQual_;
      };

      class BMTFUnpackerInputsNewQual : public Unpacker
      {
         public:
            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
         private:
            std::map<int, qualityHits> linkAndQual_;
      };

   }
}

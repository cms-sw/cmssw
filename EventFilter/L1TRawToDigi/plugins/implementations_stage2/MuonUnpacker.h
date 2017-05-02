#ifndef L1T_PACKER_STAGE2_MUONUNPACKER_H
#define L1T_PACKER_STAGE2_MUONUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
   namespace stage2 {
      class MuonUnpacker : public Unpacker {
         public:
            MuonUnpacker();
            ~MuonUnpacker() {};

            virtual bool unpack(const Block& block, UnpackerCollections *coll) override;

            inline unsigned int getAlgoVersion() { return algoVersion_; };
            inline int getFedNumber() { return fed_; };
            inline unsigned int getMuonCopy() { return muonCopy_; };

            inline void setAlgoVersion(const unsigned int version) { algoVersion_ = version; };
            inline void setFedNumber(const int fed) { fed_ = fed; };
            inline void setMuonCopy(const unsigned int copy) { muonCopy_ = copy; };

         private:
            unsigned int algoVersion_;
            int fed_;
            unsigned int muonCopy_;

      };
   }
}

#endif

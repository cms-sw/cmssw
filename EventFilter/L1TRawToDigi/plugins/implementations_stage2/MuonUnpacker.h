#ifndef L1T_PACKER_STAGE2_MUONUNPACKER_H
#define L1T_PACKER_STAGE2_MUONUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "L1TObjectCollections.h"

namespace l1t {
   namespace stage2 {
      class MuonUnpacker : public Unpacker {
         public:
            MuonUnpacker();
            ~MuonUnpacker() override {};

            bool unpack(const Block& block, UnpackerCollections *coll) override;

            inline int getFedNumber() { return fed_; };
            inline unsigned int getMuonCopy() { return muonCopy_; };

            inline void setFedNumber(const int fed) { fed_ = fed; };
            inline void setMuonCopy(const unsigned int copy) { muonCopy_ = copy; };

         private:
	    static constexpr unsigned nWords_ = 6; // every link transmits 6 words (3 muons) per bx
            static constexpr unsigned bxzs_enable_shift_ = 1;

            MuonBxCollection* res_;
            int fed_;
            unsigned int muonCopy_;

            void unpackBx(int bx, const std::vector<uint32_t>& payload, unsigned int startIdx=0);
      };
   }
}

#endif

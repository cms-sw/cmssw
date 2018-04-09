#ifndef EventFilter_L1TRawToDigi_Unpacker_h
#define EventFilter_L1TRawToDigi_Unpacker_h

#include "EventFilter/L1TRawToDigi/interface/Block.h"

namespace l1t {
   class UnpackerCollections;

   void getBXRange(int nbx, int& first, int& last);

   class Unpacker {
      public:
         virtual bool unpack(const Block& block, UnpackerCollections *coll) = 0;
         virtual ~Unpacker() = default;
      
         // Modeled on plugins/implementations_stage2/MuonUnpacker.h
         inline unsigned int getAlgoVersion() { return algoVersion_; };
         inline void setAlgoVersion(const unsigned int version) { algoVersion_ = version; };
      
      private:
         unsigned int algoVersion_;
   };
}

#endif

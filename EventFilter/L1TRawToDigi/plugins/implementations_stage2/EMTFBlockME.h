#ifndef L1T_PACKER_STAGE2_EMTFBLOCKMEUNPACKER_H
#define L1T_PACKER_STAGE2_EMTFBLOCKMEUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class MEBlockUnpacker : public Unpacker { // "MEBlockUnpacker" inherits from "Unpacker"
      public:
	virtual int  checkFormat(const Block& block);
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class MEBlockPacker : public Packer { // "MEBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

#endif

#ifndef L1T_PACKER_STAGE2_EMTFBLOCKTRAILERSUNPACKER_H
#define L1T_PACKER_STAGE2_EMTFBLOCKTRAILERSUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class TrailersBlockUnpacker : public Unpacker { // "TrailersBlockUnpacker" inherits from "Unpacker"
      public:
	virtual int  checkFormat(const Block& block);
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class TrailersBlockPacker : public Packer { // "TrailersBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

#endif

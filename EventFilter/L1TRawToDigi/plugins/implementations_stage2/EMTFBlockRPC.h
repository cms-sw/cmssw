#ifndef L1T_PACKER_STAGE2_EMTFBLOCKRPCUNPACKER_H
#define L1T_PACKER_STAGE2_EMTFBLOCKRPCUNPACKER_H

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class RPCBlockUnpacker : public Unpacker { // "RPCBlockUnpacker" inherits from "Unpacker"
      public:
	virtual int  checkFormat(const Block& block); 
	// virtual bool checkFormat() override; // Return "false" if block format does not match expected format
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class RPCBlockPacker : public Packer { // "RPCBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

#endif

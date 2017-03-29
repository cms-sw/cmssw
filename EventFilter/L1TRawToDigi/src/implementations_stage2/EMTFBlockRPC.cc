// Code to unpack the "RPC Data Record"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockRPC.h file is needed
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

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      int RPCBlockUnpacker::checkFormat(const Block& block) { 				
	
	auto payload = block.payload();
	int errors = 0;
	
	//Check the number of 16-bit words
	if(payload.size() != 4) { errors += 1; edm::LogError("L1T|EMTF") << "Payload size in 'RPC Data Record' is different than expected"; }
	
	//Check that each word is 16 bits
	if(GetHexBits(payload[0], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[0] has more than 16 bits in 'RPC Data Record'"; }
	if(GetHexBits(payload[1], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[1] has more than 16 bits in 'RPC Data Record'"; }
	if(GetHexBits(payload[2], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[2] has more than 16 bits in 'RPC Data Record'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'RPC Data Record'"; }
	
	uint16_t RPCa = payload[0];
	uint16_t RPCb = payload[1];
	uint16_t RPCc = payload[2];
	uint16_t RPCd = payload[3];
	
	//Check Format
	if(GetHexBits(RPCa, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in RPCa are incorrect"; }
	if(GetHexBits(RPCb, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in RPCb are incorrect"; }
	if(GetHexBits(RPCc, 12, 13) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in RPCc are incorrect"; }
	if(GetHexBits(RPCc, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in RPCc are incorrect"; }
	if(GetHexBits(RPCd, 3, 15)  != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in RPCd are incorrect"; }

	return errors;
	
      }     
      
      bool RPCBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// std::cout << "Inside EMTFBlockRPC.cc: unpack" << std::endl;
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();
	
	// Check Format of Payload
	l1t::emtf::RPC RPC_;	
	for (int err = 0; err < checkFormat(block); err++) RPC_.add_format_error();
	
	// Assign payload to 16-bit words
	uint16_t RPCa = payload[0];
	uint16_t RPCb = payload[1];
	uint16_t RPCc = payload[2];
	uint16_t RPCd = payload[3];
	
	// res is a pointer to a collection of EMTFDaqOut class objects
	// There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
	EMTFDaqOutCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
	int iOut = res->size() - 1;
	if (RPC_.Format_Errors() > 0) goto write;
	
	////////////////////////////
	// Unpack the RPC Data Record
	////////////////////////////
	
	RPC_.set_partition_data ( GetHexBits(RPCa,  0,  7) );
	RPC_.set_partition_num  ( GetHexBits(RPCa,  8, 11) );
	RPC_.set_prt_delay      ( GetHexBits(RPCa, 12, 14) );
	
	RPC_.set_link_number    ( GetHexBits(RPCb,  0,  4) );
	RPC_.set_lb             ( GetHexBits(RPCb,  5,  6) );
	RPC_.set_eod            ( GetHexBits(RPCb,  7,  7) );
	RPC_.set_bcn            ( GetHexBits(RPCb,  8, 14) );
	
	RPC_.set_bxn            ( GetHexBits(RPCc,  0, 11) );
	RPC_.set_bc0            ( GetHexBits(RPCc, 14, 14) );
	
	RPC_.set_tbin           ( GetHexBits(RPCd,  0,  2) );
	
	// RPC_.set_dataword            ( uint64_t dataword);

      write:
	
	(res->at(iOut)).push_RPC(RPC_);
	
	// Finished with unpacking one RPC Data Record
	return true;
	
      } // End bool RPCBlockUnpacker::unpack
      
      // bool RPCBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside RPCBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool RPCBlockPacker::pack
      
    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::RPCBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::RPCBlockPacker);

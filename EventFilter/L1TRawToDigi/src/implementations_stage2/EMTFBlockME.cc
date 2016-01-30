// Code to unpack the "ME Data Record"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockME.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class MEBlockUnpacker : public Unpacker { // "MEBlockUnpacker" inherits from "Unpacker"
      public:
	// virtual bool checkFormat() override; // Return "false" if block format does not match expected format
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

namespace l1t {
  namespace stage2 {
    namespace emtf {

      EMTFUnpackerTools tools3;

      bool MEBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();

	// std::cout << "This payload has " << payload.size() << " 16-bit words" << std::endl;
	// for (uint iWord = 0; iWord < payload.size(); iWord++)
	//   std::cout << std::hex << std::setw(8) << std::setfill('0') << payload[iWord] << std::dec << std::endl;

	// res is a pointer to a collection of EMTFOutput class objects
	// There is one EMTFOutput for each MTF7 (60 deg. sector) in the event
	EMTFOutputCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFOutputs();
	int iOut = res->size() - 1;

	////////////////////////////
	// Unpack the ME Data Record
	////////////////////////////

	l1t::emtf::ME ME_;

	ME_.set_me_bxn              ( tools3.GetHexBits(payload[2], 0, 11) );
	ME_.set_key_wire_group      ( tools3.GetHexBits(payload[0], 8, 14) );
	ME_.set_clct_key_half_strip ( tools3.GetHexBits(payload[1], 0, 7) );
	// ME_.set_quality             ( tools3.GetHexBits(payload[], , ) );
	ME_.set_clct_pattern        ( tools3.GetHexBits(payload[0], 0, 3) );
	ME_.set_id                  ( tools3.GetHexBits(payload[1], 8, 10) );
	// ME_.set_epc                 ( tools3.GetHexBits(payload[], , ) );
	ME_.set_station             ( tools3.GetHexBits(payload[3], 4, 6) );
	// ME_.set_tbin_num            ( tools3.GetHexBits(payload[], , ) );
	ME_.set_bc0                 ( tools3.GetHexBits(payload[1], 14, 14) );
	ME_.set_bxe                 ( tools3.GetHexBits(payload[1], 13, 13) );
	ME_.set_lr                  ( tools3.GetHexBits(payload[1], 12, 12) );
	// ME_.set_afff                ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_cik                 ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_nit                 ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_afef                ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_se                  ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_sm                  ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_af                  ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_vp                  ( tools3.GetHexBits(payload[], , ) );
	// ME_.set_dataword            ( uint64_t dataword);

	(res->at(iOut)).push_ME(ME_);

	// Finished with unpacking one ME Data Record
	return true;
	
      } // End bool MEBlockUnpacker::unpack

      // bool MEBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside MEBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool MEBlockPacker::pack

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::MEBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::MEBlockPacker);

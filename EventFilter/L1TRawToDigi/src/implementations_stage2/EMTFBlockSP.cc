// Code to unpack the "SP Output Data Record"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockSP.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class SPBlockUnpacker : public Unpacker { // "SPBlockUnpacker" inherits from "Unpacker"
      public:
	// virtual bool checkFormat() override; // Return "false" if block format does not match expected format
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class SPBlockPacker : public Packer { // "SPBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

namespace l1t {
  namespace stage2 {
    namespace emtf {

      EMTFUnpackerTools tools4;

      bool SPBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
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

	///////////////////////////////////
	// Unpack the SP Output Data Record
	///////////////////////////////////

	l1t::emtf::SP SP_;

	// SP_.set_pt_lut_address ( tools4.GetHexBits(payload[], , ) );
	SP_.set_phi_full       ( tools4.GetHexBits(payload[0], 0, 11) );
	SP_.set_phi_gmt        ( tools4.GetHexBits(payload[1], 0, 7) );
	SP_.set_eta            ( tools4.GetHexBits(payload[2], 0, 8) );
	SP_.set_pt             ( tools4.GetHexBits(payload[3], 0, 8) );
	SP_.set_quality        ( tools4.GetHexBits(payload[2], 9, 12) );
	SP_.set_bx             ( tools4.GetHexBits(payload[2], 13, 14) );
	SP_.set_me4_id         ( tools4.GetHexBits(payload[4], 10, 14) );
	SP_.set_me3_id         ( tools4.GetHexBits(payload[4], 5, 9) );
	SP_.set_me2_id         ( tools4.GetHexBits(payload[4], 0, 4) );
	SP_.set_me1_id         ( tools4.GetHexBits(payload[3], 9, 14) );
	// SP_.set_me4_tbin       ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_me3_tbin       ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_me2_tbin       ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_me1_tbin       ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_tbin_num       ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_hl             ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_c              ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_vc             ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_vt             ( tools4.GetHexBits(payload[], , ) );
	// SP_.set_se             ( tools4.GetHexBits(payload[], , ) );
	SP_.set_bc0            ( tools4.GetHexBits(payload[1], 12, 12) );
	// SP.set_dataword        ( uint64_t dataword );

	(res->at(iOut)).push_SP(SP_);

	// Finished with unpacking one SP Output Data Record
	return true;
	
      } // End bool SPBlockUnpacker::unpack

      // bool SPBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside SPBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool SPBlockPacker::pack

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::SPBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::SPBlockPacker);

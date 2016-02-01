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

      bool SPBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();

        // Assign payload to 16-bit words
        uint16_t SP1a = payload[0];	
        uint16_t SP1b = payload[1];	
        uint16_t SP1c = payload[2];	
        uint16_t SP1d = payload[3];	
        uint16_t SP2a = payload[4];	
        uint16_t SP2b = payload[5];	
        uint16_t SP2c = payload[6];	
        uint16_t SP2d = payload[7];	

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

	SP_.set_phi_full       ( GetHexBits(SP1a,  0, 11) );
	SP_.set_vc             ( GetHexBits(SP1a, 12, 12) );
	SP_.set_c              ( GetHexBits(SP1a, 13, 13) );
	SP_.set_hl             ( GetHexBits(SP1a, 14, 14) );

	SP_.set_phi_GMT        ( GetHexBits(SP1b,  0,  7) );
	SP_.set_bc0            ( GetHexBits(SP1b, 12, 12) );
	SP_.set_se             ( GetHexBits(SP1b, 13, 13) );
	SP_.set_vt             ( GetHexBits(SP1b, 14, 14) );

	SP_.set_eta_GMT        ( GetHexBits(SP1c,  0,  8) );
	SP_.set_quality        ( GetHexBits(SP1c,  9, 12) );
	SP_.set_bx             ( GetHexBits(SP1c, 13, 14) );

	SP_.set_pt             ( GetHexBits(SP1d,  0,  8) );
	SP_.set_me1_ID         ( GetHexBits(SP1d,  9, 14) );

	SP_.set_me2_ID         ( GetHexBits(SP2a,  0,  4) );
	SP_.set_me3_ID         ( GetHexBits(SP2a,  5,  9) );
	SP_.set_me4_ID         ( GetHexBits(SP2a, 10, 14) );

	SP_.set_me1_TBIN       ( GetHexBits(SP2b,  0,  2) );
	SP_.set_me2_TBIN       ( GetHexBits(SP2b,  3,  5) );
	SP_.set_me3_TBIN       ( GetHexBits(SP2b,  6,  8) );
	SP_.set_me4_TBIN       ( GetHexBits(SP2b,  9, 11) );
	SP_.set_TBIN_num       ( GetHexBits(SP2b, 12, 14) );

	SP_.set_pt_lut_address ( GetHexBits(SP2c,  0, 14, SP2d,  0, 14) );

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

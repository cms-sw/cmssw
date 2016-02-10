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
	virtual int  checkFormat(const Block& block);
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

      int SPBlockUnpacker::checkFormat(const Block& block) {

	auto payload = block.payload();
	int errors = 0;

	//Check the number of 16-bit words                                                                                                                                    
	if(payload.size() != 8) { errors += 1; edm::LogError("L1T|EMTF") << "Payload size in 'SP Output Data Record' is different than expected"; }

	//Check that each word is 16 bits                                                                                                                                     
	if(GetHexBits(payload[0], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[0] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[1], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[1] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[2], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[2] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[4], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[4] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[5], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[5] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[6], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[6] has more than 16 bits in 'SP Output Data Record'"; }
	if(GetHexBits(payload[7], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[7] has more than 16 bits in 'SP Output Data Record'"; }

	uint16_t SP1a = payload[0];
	uint16_t SP1b = payload[1];
	uint16_t SP1c = payload[2];
	uint16_t SP1d = payload[3];
	uint16_t SP2a = payload[4];
	uint16_t SP2b = payload[5];
	uint16_t SP2c = payload[6];
	uint16_t SP2d = payload[7];
      

	//Check Format                                                                                                                                                        
	if(GetHexBits(SP1a, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1a are incorrect"; }
	if(GetHexBits(SP1b, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1b are incorrect"; }
	if(GetHexBits(SP1b, 8, 11)  != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1b are incorrect"; }
	if(GetHexBits(SP1c, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1c are incorrect"; }
	if(GetHexBits(SP1d, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1d are incorrect"; }
	if(GetHexBits(SP2a, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2a are incorrect"; }
	if(GetHexBits(SP2b, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2b are incorrect"; }
	if(GetHexBits(SP2c, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2c are incorrect"; }
	if(GetHexBits(SP2d, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2d are incorrect"; }

	return errors;

      }


      bool SPBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();

	// Check Format of Payload
	l1t::emtf::SP SP_;
	for (int err = 0; err < checkFormat(block); err++) SP_.add_format_error();

        // Assign payload to 16-bit words
        uint16_t SP1a = payload[0];	
        uint16_t SP1b = payload[1];	
        uint16_t SP1c = payload[2];	
        uint16_t SP1d = payload[3];	
        uint16_t SP2a = payload[4];	
        uint16_t SP2b = payload[5];	
        uint16_t SP2c = payload[6];	
        uint16_t SP2d = payload[7];	

	// res is a pointer to a collection of EMTFOutput class objects
	// There is one EMTFOutput for each MTF7 (60 deg. sector) in the event
	EMTFOutputCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFOutputs();
	int iOut = res->size() - 1;

	RegionalMuonCandBxCollection* res_cand;
	res_cand = static_cast<EMTFCollections*>(coll)->getRegionalMuonCands();
	RegionalMuonCand mu_;
	
	if (SP_.Format_Errors() > 0) goto write;

	///////////////////////////////////
	// Unpack the SP Output Data Record
	///////////////////////////////////

	SP_.set_phi_full_int   ( GetHexBits(SP1a,  0, 11) );
	SP_.set_vc             ( GetHexBits(SP1a, 12, 12) );
	SP_.set_c              ( GetHexBits(SP1a, 13, 13) );
	SP_.set_hl             ( GetHexBits(SP1a, 14, 14) );

	SP_.set_phi_GMT_int    ( TwosCompl(8, GetHexBits(SP1b, 0, 7)) );
	mu_.setHwPhi           ( TwosCompl(8, GetHexBits(SP1b, 0, 7)) );
	SP_.set_bc0            ( GetHexBits(SP1b, 12, 12) );
	SP_.set_se             ( GetHexBits(SP1b, 13, 13) );
	SP_.set_vt             ( GetHexBits(SP1b, 14, 14) );

	SP_.set_eta_GMT_int    ( TwosCompl(9, GetHexBits(SP1c, 0, 8)) );
	mu_.setHwEta           ( TwosCompl(9, GetHexBits(SP1c, 0, 8)) );
	SP_.set_quality        ( GetHexBits(SP1c,  9, 12) );
	mu_.setHwQual          ( GetHexBits(SP1c,  9, 12) );
	SP_.set_bx             ( GetHexBits(SP1c, 13, 14) );

	SP_.set_pt_int         ( GetHexBits(SP1d,  0,  8) );
	mu_.setHwPt            ( GetHexBits(SP1d,  0,  8) );
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

	// SP_.set_dataword        ( uint64_t dataword );
	// mu_.set_dataword        ( uint64_t dataword );

      write:

	(res->at(iOut)).push_SP(SP_);

	res_cand->setBXRange(0, 0);
	res_cand->push_back(0, mu_);

	// int iOut_cand = res_cand->size(0) - 1;
	// (res_cand->at(iOut_cand)).setBXRange(0, 0);
	// (res_cand->at(iOut_cand)).push_back(0, mu_);

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

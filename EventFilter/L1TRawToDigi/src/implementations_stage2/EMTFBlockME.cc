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

namespace l1t {
  namespace stage2 {
    namespace emtf {

      int MEBlockUnpacker::checkFormat(const Block& block) {
	
	auto payload = block.payload();
	int errors = 0;
	
	//Check the number of 16-bit words
	if(payload.size() != 4) { errors += 1; edm::LogError("L1T|EMTF") << "Payload size in 'ME Data Record' is different than expected"; }
	
	//Check that each word is 16 bits
	if(GetHexBits(payload[0], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[0] has more than 16 bits in 'ME Data Record'"; }
	if(GetHexBits(payload[1], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[1] has more than 16 bits in 'ME Data Record'"; }
	if(GetHexBits(payload[2], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[2] has more than 16 bits in 'ME Data Record'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'ME Data Record'"; }
	
	uint16_t MEa = payload[0];
	uint16_t MEb = payload[1];
	uint16_t MEc = payload[2];
	uint16_t MEd = payload[3];

	//Check Format
	if(GetHexBits(MEa, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in MEa are incorrect"; }
	if(GetHexBits(MEb, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in MEb are incorrect"; }
	if(GetHexBits(MEc, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in MEc are incorrect"; }
	if(GetHexBits(MEd, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in MEd are incorrect"; }

	return errors;

      }


      bool MEBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {

	// std::cout << "Inside EMTFBlockME.cc: unpack" << std::endl;
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();

	// Assign payload to 16-bit words
        uint16_t MEa = payload[0];
        uint16_t MEb = payload[1];
        uint16_t MEc = payload[2];
        uint16_t MEd = payload[3];

	// Check Format of Payload
	l1t::emtf::ME ME_;
	for (int err = 0; err < checkFormat(block); err++) ME_.add_format_error();

	// res is a pointer to a collection of EMTFOutput class objects
	// There is one EMTFOutput for each MTF7 (60 deg. sector) in the event
	EMTFOutputCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFOutputs();
	int iOut = res->size() - 1;
	std::vector<int> conv_vals_ME;
	// if (ME_.Format_Errors() > 0) goto write; // Temporarily disable for DQM operation - AWB 09.04.16

	////////////////////////////
	// Unpack the ME Data Record
	////////////////////////////

	ME_.set_clct_pattern ( GetHexBits(MEa,  0,  3) );
	ME_.set_quality      ( GetHexBits(MEa,  4,  7) );
	ME_.set_wire         ( GetHexBits(MEa,  8, 14) );

	ME_.set_strip        ( GetHexBits(MEb,  0,  7) );
	ME_.set_csc_ID       ( GetHexBits(MEb,  8, 11) );
	ME_.set_lr           ( GetHexBits(MEb, 12, 12) );
	ME_.set_bxe          ( GetHexBits(MEb, 13, 13) );
	ME_.set_bc0          ( GetHexBits(MEb, 14, 14) );

	ME_.set_me_bxn       ( GetHexBits(MEc,  0, 11) );
	ME_.set_nit          ( GetHexBits(MEc, 12, 12) );
	ME_.set_cik          ( GetHexBits(MEc, 13, 13) );
	ME_.set_afff         ( GetHexBits(MEc, 14, 14) );

	ME_.set_tbin_num     ( GetHexBits(MEd,  0,  2) );
	ME_.set_vp           ( GetHexBits(MEd,  3,  3) );
	ME_.set_station      ( GetHexBits(MEd,  4,  6) );
	ME_.set_af           ( GetHexBits(MEd,  7,  7) );
	ME_.set_epc          ( GetHexBits(MEd,  8, 11) );
	ME_.set_sm           ( GetHexBits(MEd, 12, 12) );
	ME_.set_se           ( GetHexBits(MEd, 13, 13) );
	ME_.set_afef         ( GetHexBits(MEd, 14, 14) );

	// ME_.set_dataword     ( uint64_t dataword);

	// After 01.04.16
	// std::cout << "About to convert values station = " << ME_.Station() << ", CSC ID = " 
	// 	  << ME_.CSC_ID() << ", sector = " << (res->at(iOut)).PtrEventHeader()->Sector() << std::endl;
	conv_vals_ME = ME_.convert_chamber_ME( ME_.Station(), ME_.CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector() );
	ME_.set_station   ( conv_vals_ME.at(0) );
	ME_.set_csc_ID    ( conv_vals_ME.at(1) );
	ME_.set_sector    ( conv_vals_ME.at(2) );
	ME_.set_subsector ( conv_vals_ME.at(3) );
	ME_.set_neighbor  ( conv_vals_ME.at(4) );
	// std::cout << "Got converted values station = " << ME_.Station() << ", CSC ID = " << ME_.CSC_ID() 
	// 	  << ", sector = " << ME_.Sector() << ", subsector = " << ME_.Subsector() << ", neighbor = " << ME_.Neighbor() << std::endl;

	ME_.set_ring ( ME_.calc_ring_ME( ME_.Station(), ME_.CSC_ID(), ME_.Strip() ) );
	// std::cout << "From strip = " << ME_.Strip() << " got ring = " << ME_.Ring() << std::endl;

	if ( (res->at(iOut)).NumME() > 0 )
	  if (ME_.Station() == (res->at(iOut)).PtrMECollection()->at( (res->at(iOut)).NumME() - 1 ).Station() )
	    ME_.set_stub_num( (res->at(iOut)).PtrMECollection()->at( (res->at(iOut)).NumME() - 1 ).Stub_num() + 1 );
	  else ME_.set_stub_num(0);
	else ME_.set_stub_num(0);
	if (ME_.Station() < 0) ME_.set_stub_num(-99);
	      

	// write: // Temporarily disable for DQM operation - AWB 09.04.16

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

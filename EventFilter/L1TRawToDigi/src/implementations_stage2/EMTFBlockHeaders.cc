// Code to unpack the AMC13 header, "AMC data header", and "Event Record Header"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockHeaders.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class HeadersBlockUnpacker : public Unpacker { // "HeadersBlockUnpacker" inherits from "Unpacker"
      public:
	// virtual bool checkFormat() override; // Return "false" if block format does not match expected format
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class HeadersBlockPacker : public Packer { // "HeadersBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

namespace l1t {
  namespace stage2 {
	  namespace emtf {

		  void checkFormat(auto payload){
			  //Check the number of 16-bit words
			  if(payload.size() != 12) edm::LogError("L1T|EMTF") << "Payload size in 'AMC Data Header' is different than expected";

			  //Check that each word is 16 bits
			  if(GetHexBits(payload[0], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[0] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[1], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[1] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[2], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[2] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[3], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[3] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[4], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[4] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[5], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[5] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[6], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[6] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[7], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[7] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[8], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[8] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[9], 16, 31) != 0) edm::LogError("L1T|EMTF")  << "Payload[9] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[10], 16, 31) != 0) edm::LogError("L1T|EMTF") << "Payload[10] has more than 16 bits in 'AMC Data Header'";
			  if(GetHexBits(payload[11], 16, 31) != 0) edm::LogError("L1T|EMTF") << "Payload[11] has more than 16 bits in 'AMC Data Header'";

			  uint16_t hd1a = payload[0];
			  uint16_t hd1b = payload[1];
			  uint16_t hd1c = payload[2];
			  uint16_t hd1d = payload[3];
			  uint16_t hd2a = payload[4];
			  uint16_t hd2b = payload[5];
			  uint16_t hd2c = payload[6];
			  uint16_t hd2d = payload[7];
			  uint16_t hd3a = payload[8];
			  uint16_t hd3b = payload[9];
			  uint16_t hd3c = payload[10];
			  uint16_t hd3d = payload[11];

			  //Check Format
			  if(GetHexBits(hd1a, 12, 15) != 9) edm::LogError("L1T|EMTF") << "Format identifier bits in HD1a are incorrect";
			  if(GetHexBits(hd1b, 12, 15) != 9) edm::LogError("L1T|EMTF") << "Format identifier bits in HD1b are incorrect";
			  if(GetHexBits(hd1c, 12, 15) != 9) edm::LogError("L1T|EMTF") << "Format identifier bits in HD1c are incorrect";
			  if(GetHexBits(hd1c, 0, 11)  != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD1c are incorrect";
			  if(GetHexBits(hd1d, 12, 15) != 9) edm::LogError("L1T|EMTF") << "Format identifier bits in HD1d are incorrect";
			  if(GetHexBits(hd2a, 12, 15) != 10) edm::LogError("L1T|EMTF") << "Format identifier bits in HD2a are incorrect";
			  if(GetHexBits(hd2a, 0, 11)  != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD2a are incorrect";
			  if(GetHexBits(hd2b, 12, 15) != 10) edm::LogError("L1T|EMTF") << "Format identifier bits in HD2b are incorrect";
			  if(GetHexBits(hd2c, 12, 15) != 10) edm::LogError("L1T|EMTF") << "Format identifier bits in HD2c are incorrect";
			  if(GetHexBits(hd2c, 11, 11) != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD2c are incorrect";
			  if(GetHexBits(hd2d, 12, 15) != 10) edm::LogError("L1T|EMTF") << "Format identifier bits in HD2d are incorrect";
			  if(GetHexBits(hd3a, 9, 14) != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD3a are incorrect";
			  if(GetHexBits(hd3a, 15, 15) != 1) edm::LogError("L1T|EMTF") << "Format identifier bits in HD3a are incorrect";
			  if(GetHexBits(hd3b, 11, 15) != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD3b are incorrect";
			  if(GetHexBits(hd3c, 11, 15) != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD3c are incorrect";
			  if(GetHexBits(hd3d, 11, 15) != 0) edm::LogError("L1T|EMTF") << "Format identifier bits in HD3d are incorrect";
                                        
		  }       

      bool HeadersBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();

	///////////////////////////////
	// Check Format of Payload
	///////////////////////////////

	checkFormat(payload);

	// TODO: Proper error handling for payload size check (also in other Block functions)
	// std::cout << "This payload has " << payload.size() << " 16-bit words" << std::endl;
	if (payload.size() != 12) {
	  std::cout << "Critical error in EMTFBlockHeaders.cc: payload.size() = " 
		    << payload.size() << ", not 12!!!" << std::endl;
	  // return 0;
	}

	// TODO: Proper error handling for > 16-bit words (also in other Block functions)
	for (uint iWord = 0; iWord < payload.size(); iWord++) {
	  // std::cout << std::hex << std::setw(4) << std::setfill('0') << payload[iWord] << std::dec << std::endl;
	  if ( (payload[iWord] >> 16) > 0 ) {
	    std::cout << "Critical error: payload[" << iWord << "] = " << std::hex << payload[iWord]
		      << std::dec << ", more than 16 bits!!!" << std::dec << std::endl;
	    // return 0; 
	  }
	}

	// Assign payload to 16-bit words
	uint16_t HD1a = payload[0];
	uint16_t HD1b = payload[1];
	// uint16_t HD1c = payload[2];
	uint16_t HD1d = payload[3];
	// uint16_t HD2a = payload[4];
	uint16_t HD2b = payload[5];
	uint16_t HD2c = payload[6];
	uint16_t HD2d = payload[7];
	uint16_t HD3a = payload[8];
	uint16_t HD3b = payload[9];
	uint16_t HD3c = payload[10];
	uint16_t HD3d = payload[11];

	// TODO: Proper checks for Event Record Header format (also in other Block functions)

	// res is a pointer to a collection of EMTFOutput class objects
	// There is one EMTFOutput for each MTF7 (60 deg. sector) in the event
	EMTFOutputCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFOutputs();
	
	EMTFOutput EMTFOutput_;
	// std::cout << "So far " << res->size() << " MTF7s have been unpacked" << std::endl;
	res->push_back(EMTFOutput_);
	int iOut = res->size() - 1;

	//////////////////////////////////////
	// Unpack the AMC13 header information
	//////////////////////////////////////
	
	if ( (res->at(iOut)).HasAMC13Header() == true )
	  std::cout << "Why is there already an AMC13Header?" << std::endl;
	l1t::emtf::AMC13Header AMC13Header_;

	// TODO: Write functions in interface/AMC13Spec.h (as in AMCSpec.h) to extract all AMC13 header and trailer info
	// TODO: Edit interface/Block.h to have a amc13() function similar to amc()

	// AMC13Header_.set_orn( block.amc13().get() )          {  orn = bits; };
	// AMC13Header_.set_lv1_id( block.amc13().get() )       {  lv1_id = bits; };
	// AMC13Header_.set_bx_id( block.amc13().get() )        {  bx_id = bits; };
	// AMC13Header_.set_source_id( block.amc13().get() )    {  source_id = bits; };
	// AMC13Header_.set_evt_ty( block.amc13().get() )       {  evt_ty = bits; };
	// AMC13Header_.set_fov( block.amc13().get() )          {  fov = bits; };
	// AMC13Header_.set_ufov( block.amc13().get() )         {  ufov = bits; };
	// AMC13Header_.set_res( block.amc13().get() )          {  res = bits; };
	// AMC13Header_.set_namc( block.amc13().get() )         {  namc = bits; };
	// AMC13Header_.set_h( block.amc13().get() )            {  h = bits; };
	// AMC13Header_.set_x( block.amc13().get() )            {  x = bits; };
	// AMC13Header_.set_dataword(uint64_t bits)  { dataword = bits; };
	
	(res->at(iOut)).set_AMC13Header(AMC13Header_);
	
	/////////////////////////////////////
	// Unpack the MTF7 header information
	/////////////////////////////////////

	if ( (res->at(iOut)).HasMTF7Header() == true )
	  std::cout << "Why is there already an MTF7Header?" << std::endl;
	l1t::emtf::MTF7Header MTF7Header_;

	// AMC header info defined in interface/AMCSpec.h
	MTF7Header_.set_amc_number   ( block.amc().getAMCNumber() );
	MTF7Header_.set_bx_id        ( block.amc().getBX() );
	MTF7Header_.set_orbit_number ( block.amc().getOrbitNumber() );
	MTF7Header_.set_board_id     ( block.amc().getBoardID() );
	MTF7Header_.set_lv1_id       ( block.amc().getLV1ID() );
	MTF7Header_.set_data_length  ( block.amc().getSize() );
	MTF7Header_.set_user_id      ( block.amc().getUserData() );
	// MTF7Header_.set_dataword(uint64_t bits)  { dataword = bits;    };	
	
	(res->at(iOut)).set_MTF7Header(MTF7Header_);

	/////////////////////////////////////////////
	// Unpack the Event Record header information
	/////////////////////////////////////////////
	
	if ( (res->at(iOut)).HasEventHeader() == true )
	  std::cout << "Why is there already an EventHeader?" << std::endl;
	l1t::emtf::EventHeader EventHeader_;
	
	EventHeader_.set_l1a     ( GetHexBits(HD1a,  0, 11, HD1b,  0, 11) );
	EventHeader_.set_l1a_bxn ( GetHexBits(HD1d,  0, 11) );
	EventHeader_.set_sp_ts   ( GetHexBits(HD2b,  8, 11) );
	EventHeader_.set_sp_ersv ( GetHexBits(HD2b,  5,  7) );
	EventHeader_.set_sp_addr ( GetHexBits(HD2b,  0,  4) );
	EventHeader_.set_tbin    ( GetHexBits(HD2c,  8, 10) );
	EventHeader_.set_ddm     ( GetHexBits(HD2c,  7,  7) );
	EventHeader_.set_spa     ( GetHexBits(HD2c,  6,  6) );
	EventHeader_.set_rpca    ( GetHexBits(HD2c,  5,  5) );
	EventHeader_.set_skip    ( GetHexBits(HD2c,  4,  4) );
	EventHeader_.set_rdy     ( GetHexBits(HD2c,  3,  3) );
	EventHeader_.set_bsy     ( GetHexBits(HD2c,  2,  2) );
	EventHeader_.set_osy     ( GetHexBits(HD2c,  1,  1) );
	EventHeader_.set_wof     ( GetHexBits(HD2c,  0,  0) );
	EventHeader_.set_me1a    ( GetHexBits(HD2d,  0, 11) );
	EventHeader_.set_me1b    ( GetHexBits(HD3a,  0,  8) );
	EventHeader_.set_me2     ( GetHexBits(HD3b,  0, 10) );
	EventHeader_.set_me3     ( GetHexBits(HD3c,  0, 10) );
	EventHeader_.set_me4     ( GetHexBits(HD3d,  0, 10) );
	// EventHeader_.set_dataword(uint64_t bits)  { dataword = bits;  };

	(res->at(iOut)).set_EventHeader(EventHeader_);

	// Finished with unpacking headers
	return true;
	
      } // End bool HeadersBlockUnpacker::unpack

      // bool HeadersBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside HeadersBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool HeadersBlockPacker::pack

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::HeadersBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::HeadersBlockPacker);

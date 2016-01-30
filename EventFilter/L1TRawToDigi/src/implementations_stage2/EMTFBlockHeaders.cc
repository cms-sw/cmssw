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

      EMTFUnpackerTools tools1;

      bool HeadersBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
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
	
	// EventHeader_.set_l1a( tools1.GetHexBits(payload[], , ) )             {  l1a = bits; };
	EventHeader_.set_l1a_bxn ( tools1.GetHexBits(payload[3],  0, 11) );
	EventHeader_.set_sp_ts   ( tools1.GetHexBits(payload[5],  8, 11) );
	EventHeader_.set_sp_ersv ( tools1.GetHexBits(payload[5],  5,  7) );
	EventHeader_.set_sp_addr ( tools1.GetHexBits(payload[5],  0,  4) );
	EventHeader_.set_tbin    ( tools1.GetHexBits(payload[6],  8, 10) );
	EventHeader_.set_ddm     ( tools1.GetHexBits(payload[6],  7,  7) );
	EventHeader_.set_spa     ( tools1.GetHexBits(payload[6],  6,  6) );
	EventHeader_.set_rpca    ( tools1.GetHexBits(payload[6],  5,  5) );
	EventHeader_.set_skip    ( tools1.GetHexBits(payload[6],  4,  4) );
	EventHeader_.set_rdy     ( tools1.GetHexBits(payload[6],  3,  3) );
	EventHeader_.set_bsy     ( tools1.GetHexBits(payload[6],  2,  2) );
	EventHeader_.set_osy     ( tools1.GetHexBits(payload[6],  1,  1) );
	EventHeader_.set_wof     ( tools1.GetHexBits(payload[6],  0,  0) );
	EventHeader_.set_me1a    ( tools1.GetHexBits(payload[7],  0, 11) );
	EventHeader_.set_me1b    ( tools1.GetHexBits(payload[8],  0,  8) );
	EventHeader_.set_me2     ( tools1.GetHexBits(payload[9],  0, 10) );
	EventHeader_.set_me3     ( tools1.GetHexBits(payload[10], 0, 10) );
	EventHeader_.set_me4     ( tools1.GetHexBits(payload[11], 0, 10) );
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

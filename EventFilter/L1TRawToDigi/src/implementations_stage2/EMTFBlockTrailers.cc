// Code to unpack the AMC13 trailer, "AMC data trailer", and "Event Record Trailer"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockTrailers.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class TrailersBlockUnpacker : public Unpacker { // "TrailersBlockUnpacker" inherits from "Unpacker"
      public:
	// virtual bool checkFormat() override; // Return "false" if block format does not match expected format
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

namespace l1t {
  namespace stage2 {
    namespace emtf {

      EMTFUnpackerTools tools5;

      bool TrailersBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
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

	/////////////////////////////////////////////
	// Unpack the Event Record trailer information
	/////////////////////////////////////////////
	
	if ( (res->at(iOut)).HasEventTrailer() == true )
	  std::cout << "Why is there already an EventTrailer?" << std::endl;
	l1t::emtf::EventTrailer EventTrailer_;

	// EventTrailer_.set_crc22     ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_ddcsr_bid ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_ddcsr_lf  ( tools5.GetHexBits(payload[], , ) );
	EventTrailer_.set_spcsr_scc ( tools5.GetHexBits(payload[3], 0, 11) );
	EventTrailer_.set_l1a       ( tools5.GetHexBits(payload[0], 0, 7) );
	// EventTrailer_.set_yy        ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_mm        ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_dd        ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_sp_ladr   ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_sp_ersv   ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_sp_padr   ( tools5.GetHexBits(payload[], , ) );
	EventTrailer_.set_lfff      ( tools5.GetHexBits(payload[1], 7, 7) );
	// EventTrailer_.set_bb        ( tools5.GetHexBits(payload[], , ) );
	// EventTrailer_.set_dataword(uint64_t bits)  { dataword = bits;  };

	(res->at(iOut)).set_EventTrailer(EventTrailer_);

	/////////////////////////////////////
	// Unpack the MTF7 trailer information
	/////////////////////////////////////

	if ( (res->at(iOut)).HasMTF7Trailer() == true )
	  std::cout << "Why is there already an MTF7Trailer?" << std::endl;
	l1t::emtf::MTF7Trailer MTF7Trailer_;

	// // AMC trailer info defined in interface/AMCSpec.h ... but not implemented in interface/Block.h?
	// MTF7Trailer_.set_crc_32( tools5.GetHexBits(payload[], , ) );
	// MTF7Trailer_.set_lv1_id( tools5.GetHexBits(payload[], , ) );
	// MTF7Trailer_.set_data_length( tools5.GetHexBits(payload[], , ) );
	// MTF7Trailer_.set_dataword(uint64_t bits)  { dataword = bits;    };
	
	(res->at(iOut)).set_MTF7Trailer(MTF7Trailer_);

	//////////////////////////////////////
	// Unpack the AMC13 trailer information
	//////////////////////////////////////
	
	if ( (res->at(iOut)).HasAMC13Trailer() == true )
	  std::cout << "Why is there already an AMC13Trailer?" << std::endl;
	l1t::emtf::AMC13Trailer AMC13Trailer_;

	// TODO: Write functions in interface/AMC13Spec.h (as in AMCSpec.h) to extract all AMC13 header and trailer info
	// TODO: Edit interface/Block.h to have a amc13() function similar to amc()

	// AMC13Trailer_.set_evt_lgth( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_crc16( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_evt_stat( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_tts( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_c( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_f( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_t( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_r( tools5.GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_dataword(uint64_t bits)  { dataword = bits; };
	
	(res->at(iOut)).set_AMC13Trailer(AMC13Trailer_);
	
	// Finished with unpacking trailers
	return true;
	
      } // End bool TrailersBlockUnpacker::unpack

      // bool TrailersBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside TrailersBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool TrailersBlockPacker::pack

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::TrailersBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::TrailersBlockPacker);

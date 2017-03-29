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
	virtual int  checkFormat(const Block& block);
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

      int TrailersBlockUnpacker::checkFormat(const Block& block) {
	
	auto payload = block.payload();
	int errors = 0;
	
	//Check the number of 16-bit words                                                                                                                                    
	if(payload.size() != 8) { errors += 1; edm::LogError("L1T|EMTF") << "Payload size in 'AMC Data Trailer' is different than expected"; }
	
	//Check that each word is 16 bits                                                                                                                                     
	if(GetHexBits(payload[0], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[0] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[1], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[1] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[2], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[2] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'AMC Data Trailer'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'AMC Data Trailer'"; }
	
	// Assign payload to 16-bit words
        uint16_t TR1a = payload[0];
        uint16_t TR1b = payload[1];
        uint16_t TR1c = payload[2];
        uint16_t TR1d = payload[3];
        uint16_t TR2a = payload[4];
        uint16_t TR2b = payload[5];
        uint16_t TR2c = payload[6];
        uint16_t TR2d = payload[7];
	
	//Check Format                                                                                                                                                        
	if(GetHexBits(TR1a, 12, 15) != 15) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1a are incorrect"; }
	if(GetHexBits(TR1b, 12, 15) != 15) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1b are incorrect"; }
	if(GetHexBits(TR1b, 0, 3)   != 15) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1b are incorrect"; }
	if(GetHexBits(TR1b, 4, 6)   != 7) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1b are incorrect"; }
	if(GetHexBits(TR1c, 9, 11)  != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1c are incorrect"; }
	// FIXME: we are consistently reading GetHexBits(TR1c, 12, 15) == 14 - AWB 10.02.16
	// if(GetHexBits(TR1c, 12, 15) != 15) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1c are incorrect"; }
	if(GetHexBits(TR1d, 12, 15) != 15) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR1d are incorrect"; }
	// FIXME: we are consistently reading GetHexBits(TR2a, 5, 11) == 18 - AWB 10.02.16
	// if(GetHexBits(TR2a, 5, 11)  != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR2a are incorrect"; }
	if(GetHexBits(TR2a, 12, 15) != 14) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR2a are incorrect"; }
	if(GetHexBits(TR2b, 12, 15) != 14) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR2b are incorrect"; }
	if(GetHexBits(TR2c, 12, 15) != 14) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR2c are incorrect"; }
	if(GetHexBits(TR2d, 12, 15) != 14) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in TR2d are incorrect"; }

	return errors;

      }

      bool TrailersBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {

	// std::cout << "Inside EMTFBlockTrailers.cc: unpack" << std::endl;
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();


	// Check Format of Payload
	l1t::emtf::AMC13Trailer AMC13Trailer_;
	l1t::emtf::MTF7Trailer MTF7Trailer_;
	l1t::emtf::EventTrailer EventTrailer_;
	for (int err = 0; err < checkFormat(block); err++) EventTrailer_.add_format_error();

	// Assign payload to 16-bit words
        uint16_t TR1a = payload[0];
        uint16_t TR1b = payload[1];
        uint16_t TR1c = payload[2];
        uint16_t TR1d = payload[3];
        uint16_t TR2a = payload[4];
        uint16_t TR2b = payload[5];
        uint16_t TR2c = payload[6];
        uint16_t TR2d = payload[7];

	// res is a pointer to a collection of EMTFDaqOut class objects
	// There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
	EMTFDaqOutCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
	int iOut = res->size() - 1;

	/////////////////////////////////////////////
	// Unpack the Event Record trailer information
	/////////////////////////////////////////////
	
	if ( (res->at(iOut)).HasEventTrailer() == true )
	  { (res->at(iOut)).add_format_error(); edm::LogError("L1T|EMTF") << "Why is there already an EventTrailer object?"; goto write_Event; }
	if (EventTrailer_.Format_Errors() > 0) goto write_Event;

	EventTrailer_.set_l1a       ( GetHexBits(TR1a,  0,  7) );
	EventTrailer_.set_ddcsr_lf  ( GetHexBits(TR1a,  8, 11, TR1b,  8, 11) );

	EventTrailer_.set_lfff      ( GetHexBits(TR1b,  7,  7) );

	// EventTrailer_.set_ddcsr_bid ( GetHexBits(TR2a,  0,  4, TR1c,  0, 8) ); // Probably incorrect
	EventTrailer_.set_mm        ( GetHexBits(TR1c,  0,  3) );
 	EventTrailer_.set_yy        ( GetHexBits(TR1c,  4,  7) );
	EventTrailer_.set_bb        ( GetHexBits(TR1c,  8,  8) );

	EventTrailer_.set_spcsr_scc ( GetHexBits(TR1d,  0, 11) );

	EventTrailer_.set_dd        ( GetHexBits(TR2a,  0,  4) );

	EventTrailer_.set_sp_padr   ( GetHexBits(TR2b,  0,  4) );
	EventTrailer_.set_sp_ersv   ( GetHexBits(TR2b,  5,  7) );
	EventTrailer_.set_sp_ladr   ( GetHexBits(TR2b,  8, 11) );

	EventTrailer_.set_crc22     ( GetHexBits(TR2c,  0, 10, TR2d,  0, 10) );
	EventTrailer_.set_lp        ( GetHexBits(TR2c, 11, 11) );
	EventTrailer_.set_hp        ( GetHexBits(TR2d, 11, 11) );

	// EventTrailer_.set_dataword(uint64_t bits)  { dataword = bits;  };

      write_Event:

	(res->at(iOut)).set_EventTrailer(EventTrailer_);

	/////////////////////////////////////
	// Unpack the MTF7 trailer information
	/////////////////////////////////////

	if ( (res->at(iOut)).HasMTF7Trailer() == true )
	  { (res->at(iOut)).add_format_error(); edm::LogError("L1T|EMTF") << "Why is there already an MTF7Trailer object?"; goto write_MTF7; }

	// // AMC trailer info defined in interface/AMCSpec.h ... but not implemented in interface/Block.h?
	// MTF7Trailer_.set_crc_32( GetHexBits(payload[], , ) );
	// MTF7Trailer_.set_lv1_id( GetHexBits(payload[], , ) );
	// MTF7Trailer_.set_data_length( GetHexBits(payload[], , ) );
	// MTF7Trailer_.set_dataword(uint64_t bits)  { dataword = bits;    };

      write_MTF7:
	
	(res->at(iOut)).set_MTF7Trailer(MTF7Trailer_);

	//////////////////////////////////////
	// Unpack the AMC13 trailer information
	//////////////////////////////////////
	
	if ( (res->at(iOut)).HasAMC13Trailer() == true )
	  { (res->at(iOut)).add_format_error(); edm::LogError("L1T|EMTF") << "Why is there already an AMC13Trailer object?"; goto write_AMC13; }

	// TODO: Write functions in interface/AMC13Spec.h (as in AMCSpec.h) to extract all AMC13 header and trailer info
	// TODO: Edit interface/Block.h to have a amc13() function similar to amc()

	// AMC13Trailer_.set_evt_lgth( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_crc16( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_evt_stat( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_tts( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_c( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_f( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_t( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_r( GetHexBits(payload[], , ) );
	// AMC13Trailer_.set_dataword(uint64_t bits)  { dataword = bits; };

      write_AMC13:
	
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

// Code to unpack the "Block of Counters"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockCounters.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class CountersBlockUnpacker : public Unpacker { // "CountersBlockUnpacker" inherits from "Unpacker"
      public:
	virtual int checkFormat(const Block& block);
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class CountersBlockPacker : public Packer { // "CountersBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      int CountersBlockUnpacker::checkFormat(const Block& block) {
	
	auto payload = block.payload();
	int errors = 0;
	
	//Check the number of 16-bit words
	if (payload.size() != 4) { 
	  errors += 1; 
	  edm::LogError("L1T|EMTF") << "Payload size in 'Block of Counters' is different than expected"; 
	}
	
	//Check that each word is 16 bits                                                                                                                                     
	if (GetHexBits(payload[0], 16, 31) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Payload[0] has more than 16 bits in 'Block of Counters'"; }
	if (GetHexBits(payload[1], 16, 31) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Payload[1] has more than 16 bits in 'Block of Counters'"; }
	if (GetHexBits(payload[2], 16, 31) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Payload[2] has more than 16 bits in 'Block of Counters'"; }
	if (GetHexBits(payload[3], 16, 31) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'Block of Counters'"; }
	
	uint16_t BCa = payload[0];
	uint16_t BCb = payload[1];
	uint16_t BCc = payload[2];
	uint16_t BCd = payload[3];
	
	//Check Format                                                                                                                                                        
	if (GetHexBits(BCa, 15, 15) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCa are incorrect"; }
	if (GetHexBits(BCb, 15, 15) != 1) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCb are incorrect"; }
	if (GetHexBits(BCc, 15, 15) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCc are incorrect"; }
	if (GetHexBits(BCd, 15, 15) != 0) { 
	  errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCd are incorrect"; }
	
	return errors;
      } // End function: int CountersBlockUnpacker::checkFormat()


      bool CountersBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// std::cout << "Inside EMTFBlockCounters.cc: unpack" << std::endl;
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();
	
	// Check Format of Payload
	l1t::emtf::Counters Counters_;
	for (int err = 0; err < checkFormat(block); err++) Counters_.add_format_error();

	// Assign payload to 16-bit words
        uint16_t BCa = payload[0];
        uint16_t BCb = payload[1];
        uint16_t BCc = payload[2];
        uint16_t BCd = payload[3];

	// res is a pointer to a collection of EMTFDaqOut class objects
	// There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
	EMTFDaqOutCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
	int iOut = res->size() - 1;

	///////////////////////////////
	// Unpack the Block of Counters
	///////////////////////////////	
	if ( (res->at(iOut)).HasCounters() == true ) { 
	  (res->at(iOut)).add_format_error(); 
	  edm::LogError("L1T|EMTF") << "Why is there already a Counters object?";
	}
	
	Counters_.set_me1a_2( GetHexBits(BCa,  0,  0) ); 
	Counters_.set_me1a_3( GetHexBits(BCa,  1,  1) ); 
	Counters_.set_me1a_4( GetHexBits(BCa,  2,  2) ); 
	Counters_.set_me1a_5( GetHexBits(BCa,  3,  3) ); 
	Counters_.set_me1a_6( GetHexBits(BCa,  4,  4) ); 
	Counters_.set_me1a_7( GetHexBits(BCa,  5,  5) ); 
	Counters_.set_me1a_8( GetHexBits(BCa,  6,  6) ); 
	Counters_.set_me1a_9( GetHexBits(BCa,  7,  7) ); 
	Counters_.set_me1b_2( GetHexBits(BCa,  8,  8) ); 
	Counters_.set_me1b_3( GetHexBits(BCa,  9,  9) ); 
	Counters_.set_me1b_4( GetHexBits(BCa, 10, 10) ); 
	Counters_.set_me1b_5( GetHexBits(BCa, 11, 11) ); 
	Counters_.set_me1b_6( GetHexBits(BCa, 12, 12) ); 
	Counters_.set_me1b_7( GetHexBits(BCa, 13, 13) ); 
	Counters_.set_me1b_8( GetHexBits(BCa, 14, 14) ); 
	
	Counters_.set_me1b_9( GetHexBits(BCb,  0,  0) ); 
	Counters_.set_me2_2 ( GetHexBits(BCb,  1,  1) ); 
	Counters_.set_me2_3 ( GetHexBits(BCb,  2,  2) ); 
	Counters_.set_me2_4 ( GetHexBits(BCb,  3,  3) ); 
	Counters_.set_me2_5 ( GetHexBits(BCb,  4,  4) ); 
	Counters_.set_me2_6 ( GetHexBits(BCb,  5,  5) ); 
	Counters_.set_me2_7 ( GetHexBits(BCb,  6,  6) ); 
	Counters_.set_me2_8 ( GetHexBits(BCb,  7,  7) ); 
	Counters_.set_me2_9 ( GetHexBits(BCb,  8,  8) ); 
	Counters_.set_me3_2 ( GetHexBits(BCb,  9,  9) ); 
	Counters_.set_me3_3 ( GetHexBits(BCb, 10, 10) ); 
	Counters_.set_me3_4 ( GetHexBits(BCb, 11, 11) ); 
	Counters_.set_me3_5 ( GetHexBits(BCb, 12, 12) ); 
	Counters_.set_me3_6 ( GetHexBits(BCb, 13, 13) ); 
	Counters_.set_me3_7 ( GetHexBits(BCb, 14, 14) ); 

	Counters_.set_me3_8 ( GetHexBits(BCc,  0,  0) ); 
	Counters_.set_me3_9 ( GetHexBits(BCc,  1,  1) ); 
	Counters_.set_me4_2 ( GetHexBits(BCc,  2,  2) ); 
	Counters_.set_me4_3 ( GetHexBits(BCc,  3,  3) ); 
	Counters_.set_me4_4 ( GetHexBits(BCc,  4,  4) ); 
	Counters_.set_me4_5 ( GetHexBits(BCc,  5,  5) ); 
	Counters_.set_me4_6 ( GetHexBits(BCc,  6,  6) ); 
	Counters_.set_me4_7 ( GetHexBits(BCc,  7,  7) ); 
	Counters_.set_me4_8 ( GetHexBits(BCc,  8,  8) ); 
	Counters_.set_me4_9 ( GetHexBits(BCc,  9,  9) ); 
	Counters_.set_me1n_3( GetHexBits(BCc, 10, 10) ); 
	Counters_.set_me1n_6( GetHexBits(BCc, 11, 11) ); 
	Counters_.set_me1n_9( GetHexBits(BCc, 12, 12) ); 
	Counters_.set_me2n_3( GetHexBits(BCc, 13, 13) ); 
	Counters_.set_me2n_9( GetHexBits(BCc, 14, 14) ); 

	Counters_.set_me3n_3( GetHexBits(BCd,  0,  0) ); 
	Counters_.set_me3n_9( GetHexBits(BCd,  1,  1) ); 
	Counters_.set_me4n_3( GetHexBits(BCd,  2,  2) ); 
	Counters_.set_me4n_9( GetHexBits(BCd,  3,  3) ); 
	Counters_.set_me1a_1( GetHexBits(BCd,  4,  4) );
	Counters_.set_me1b_1( GetHexBits(BCd,  5,  5) );
	Counters_.set_me2_1 ( GetHexBits(BCd,  6,  6) );
	Counters_.set_me3_1 ( GetHexBits(BCd,  7,  7) );
	Counters_.set_me4_1 ( GetHexBits(BCd,  8,  8) );


	Counters_.set_me1a_all( GetHexBits(BCa,  0,  7) ); 
	Counters_.set_me1b_all( GetHexBits(BCa,  8, 14, BCb, 0, 0) ); 
	Counters_.set_me2_all ( GetHexBits(BCb,  1,  8) ); 
	Counters_.set_me3_all ( GetHexBits(BCb,  9, 14, BCc, 0, 1) ); 
	Counters_.set_me4_all ( GetHexBits(BCc,  2,  9) ); 
	Counters_.set_meN_all ( GetHexBits(BCc, 10, 14, BCd, 0, 3) ); 

	Counters_.set_me1a_all( (Counters_.ME1a_all() << 1) | Counters_.ME1a_1() );
	Counters_.set_me1b_all( (Counters_.ME1b_all() << 1) | Counters_.ME1b_1() );
	Counters_.set_me2_all ( (Counters_.ME2_all()  << 1) | Counters_.ME2_1()  );
	Counters_.set_me3_all ( (Counters_.ME3_all()  << 1) | Counters_.ME3_1()  );
	Counters_.set_me4_all ( (Counters_.ME4_all()  << 1) | Counters_.ME4_1()  );

	Counters_.set_me_all  ( Counters_.MEN_all() | Counters_.ME4_all()  | Counters_.ME3_all() |
				Counters_.ME2_all() | Counters_.ME1b_all() | Counters_.ME1a_all() );

	// Counters_.set_dataword(uint64_t bits)  { dataword = bits;      };

	(res->at(iOut)).set_Counters(Counters_);
	
	// Finished with unpacking Counters
	return true;
	
      } // End bool CountersBlockUnpacker::unpack
      
      // bool CountersBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside CountersBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool CountersBlockPacker::pack

    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::CountersBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::CountersBlockPacker);

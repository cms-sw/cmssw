// Code to unpack the "Block of Counters"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"
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
	if(payload.size() != 4) { errors += 1; edm::LogError("L1T|EMTF") << "Payload size in 'Block of Counters' is different than expected"; }

	//Check that each word is 16 bits                                                                                                                                     
	if(GetHexBits(payload[0], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[0] has more than 16 bits in 'Block of Counters'"; }
	if(GetHexBits(payload[1], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[1] has more than 16 bits in 'Block of Counters'"; }
	if(GetHexBits(payload[2], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[2] has more than 16 bits in 'Block of Counters'"; }
	if(GetHexBits(payload[3], 16, 31) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Payload[3] has more than 16 bits in 'Block of Counters'"; }

	uint16_t BCa = payload[0];
	uint16_t BCb = payload[1];
	uint16_t BCc = payload[2];
	uint16_t BCd = payload[3];

	//Check Format                                                                                                                                                        
	if(GetHexBits(BCa, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCa are incorrect"; }
	if(GetHexBits(BCb, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCb are incorrect"; }
	if(GetHexBits(BCc, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCc are incorrect"; }
	if(GetHexBits(BCd, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in BCd are incorrect"; }

	return errors;
      }


      bool CountersBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
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

	// res is a pointer to a collection of EMTFOutput class objects
	// There is one EMTFOutput for each MTF7 (60 deg. sector) in the event
	EMTFOutputCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFOutputs();
	int iOut = res->size() - 1;

	///////////////////////////////
	// Unpack the Block of Counters
	///////////////////////////////	
	if ( (res->at(iOut)).HasCounters() == true )
	  { (res->at(iOut)).add_format_error(); edm::LogError("L1T|EMTF") << "Why is there already a Counters object?"; goto write; }
	if (Counters_.Format_Errors() > 0) goto write;

	Counters_.set_track_counter( GetHexBits(BCa, 0, 14, BCb, 0, 14) );
	Counters_.set_orbit_counter( GetHexBits(BCc, 0, 14, BCd, 0, 14) );
	// Counters_.set_rpc_counter( GetHexBits(payload[], , ) );
	// Counters_.set_dataword(uint64_t bits)  { dataword = bits;      };

      write:
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

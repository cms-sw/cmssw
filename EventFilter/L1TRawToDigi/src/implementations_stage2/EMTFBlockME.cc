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

      // Converts station, CSC_ID, sector, subsector, and neighbor from the ME output
      std::vector<int> convert_ME_location(int _station, int _csc_ID, int _sector) {
	int new_sector = _sector;
	int new_csc_ID = _csc_ID; // Before FW update on 05.05.16, shift by +1 from 0,1,2... convention to 1,2,3...
	if      (_station == 0) { int arr[] = {       1, new_csc_ID, new_sector,  1, 0}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (_station == 1) { int arr[] = {       1, new_csc_ID, new_sector,  2, 0}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (_station <= 4) { int arr[] = {_station, new_csc_ID, new_sector, -1, 0}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (_station == 5) new_sector = (_sector != 1) ? _sector-1 : 6;
	else { int arr[] = {_station, _csc_ID, _sector, -99, -99}; std::vector<int> vec(arr, arr+5); return vec; }
	
	if      (new_csc_ID == 1) { int arr[] = {1, 3, new_sector,  2, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 2) { int arr[] = {1, 6, new_sector,  2, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 3) { int arr[] = {1, 9, new_sector,  2, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 4) { int arr[] = {2, 3, new_sector, -1, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 5) { int arr[] = {2, 9, new_sector, -1, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 6) { int arr[] = {3, 3, new_sector, -1, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 7) { int arr[] = {3, 9, new_sector, -1, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 8) { int arr[] = {4, 3, new_sector, -1, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else if (new_csc_ID == 9) { int arr[] = {4, 9, new_sector, -1, 1}; std::vector<int> vec(arr, arr+5); return vec; }
	else                   { int arr[] = {_station, _csc_ID, _sector, -99, -99}; std::vector<int> vec(arr, arr+5); return vec; }
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

	// res is a pointer to a collection of EMTFDaqOut class objects
	// There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
	EMTFDaqOutCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
	int iOut = res->size() - 1;

	EMTFHitCollection* res_hit;
	res_hit = static_cast<EMTFCollections*>(coll)->getEMTFHits();
	EMTFHit Hit_;

	CSCCorrelatedLCTDigiCollection* res_LCT;
	res_LCT = static_cast<EMTFCollections*>(coll)->getEMTFLCTs();

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

	ME_.set_tbin         ( GetHexBits(MEd,  0,  2) );
	ME_.set_vp           ( GetHexBits(MEd,  3,  3) );
	ME_.set_station      ( GetHexBits(MEd,  4,  6) );
	ME_.set_af           ( GetHexBits(MEd,  7,  7) );
	ME_.set_epc          ( GetHexBits(MEd,  8, 11) );
	ME_.set_sm           ( GetHexBits(MEd, 12, 12) );
	ME_.set_se           ( GetHexBits(MEd, 13, 13) );
	ME_.set_afef         ( GetHexBits(MEd, 14, 14) );

	// ME_.set_dataword     ( uint64_t dataword);

	
	// Fill the EMTFHit
	Hit_.ImportME( ME_ );
	Hit_.set_endcap ( ((res->at(iOut)).PtrEventHeader()->Endcap() == 1) ? 1 : -1 );
	// Hit_.set_layer();
	
	std::vector<int> conv_vals = convert_ME_location( ME_.Station(), ME_.CSC_ID(), 
							  (res->at(iOut)).PtrEventHeader()->Sector() );
	Hit_.set_station   ( conv_vals.at(0) );
	Hit_.set_csc_ID    ( conv_vals.at(1) );
	Hit_.set_sector    ( conv_vals.at(2) );
	Hit_.set_subsector ( conv_vals.at(3) );
	Hit_.set_neighbor  ( conv_vals.at(4) );
	
	Hit_.set_sector_index ( (Hit_.Endcap() == 1) 
				? (res->at(iOut)).PtrEventHeader()->Sector() - 1
				: (res->at(iOut)).PtrEventHeader()->Sector() + 5 );

	Hit_.set_ring       ( calc_ring( Hit_.Station(), Hit_.CSC_ID(), Hit_.Strip() ) );
	Hit_.set_chamber    ( calc_chamber( Hit_.Station(), Hit_.Sector(), 
					    Hit_.Subsector(), Hit_.Ring(), Hit_.CSC_ID() ) );

	Hit_.SetCSCDetId   ( Hit_.CreateCSCDetId() );
	Hit_.SetCSCLCTDigi ( Hit_.CreateCSCCorrelatedLCTDigi() );

	// Set the stub number for this hit
	// Each chamber can send up to 2 stubs per BX
	ME_.set_stub_num(0);
	Hit_.set_stub_num(0);
	// See if matching hit is already in event record (from neighboring sector)
	bool duplicate_hit_exists = false;
	for (uint iHit = 0; iHit < res_hit->size(); iHit++) {

	  if ( Hit_.BX() == res_hit->at(iHit).BX() && Hit_.Station() == res_hit->at(iHit).Station() &&
	       ( Hit_.Ring() == res_hit->at(iHit).Ring() || abs(Hit_.Ring() - res_hit->at(iHit).Ring()) == 3 ) && 
	       Hit_.Chamber() == res_hit->at(iHit).Chamber() ) {

	    if ( Hit_.Neighbor() == res_hit->at(iHit).Neighbor() ) {
	      ME_.set_stub_num( ME_.Stub_num() + 1 );
	      Hit_.set_stub_num( Hit_.Stub_num() + 1); }
	    else if ( Hit_.Ring() == res_hit->at(iHit).Ring() && Hit_.Strip() == res_hit->at(iHit).Strip() && 
		      Hit_.Wire() == res_hit->at(iHit).Wire() )
	      duplicate_hit_exists = true;
	  }
	}
	// write: // Temporarily disable for DQM operation - AWB 09.04.16

	(res->at(iOut)).push_ME(ME_);
	res_hit->push_back(Hit_);
	if (not duplicate_hit_exists) // Don't write duplicate LCTs from adjacent sectors
	  res_LCT->insertDigi( Hit_.CSC_DetId(), Hit_.CSC_LCTDigi() );

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

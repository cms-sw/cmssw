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
        int new_csc_ID = _csc_ID; // Until FW update on 05.05.16, need to add "+1" to shift from 0,1,2... convention to 1,2,3...
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
	// if (ME_.Format_Errors() > 0) goto write; // Temporarily disable for DQM operation - AWB 09.04.16

	CSCCorrelatedLCTDigiCollection* res_LCT;
	res_LCT = static_cast<EMTFCollections*>(coll)->getEMTFLCTs();

	////////////////////////////
	// Unpack the ME Data Record
	////////////////////////////

	ME_.set_clct_pattern        ( GetHexBits(MEa,  0,  3) );
	ME_.set_quality             ( GetHexBits(MEa,  4,  7) );
	ME_.set_key_wire_group      ( GetHexBits(MEa,  8, 14) );

	ME_.set_clct_key_half_strip ( GetHexBits(MEb,  0,  7) );
	ME_.set_csc_ID              ( GetHexBits(MEb,  8, 11) );
	ME_.set_lr                  ( GetHexBits(MEb, 12, 12) );
	ME_.set_bxe                 ( GetHexBits(MEb, 13, 13) );
	ME_.set_bc0                 ( GetHexBits(MEb, 14, 14) );

	ME_.set_me_bxn              ( GetHexBits(MEc,  0, 11) );
	ME_.set_nit                 ( GetHexBits(MEc, 12, 12) );
	ME_.set_cik                 ( GetHexBits(MEc, 13, 13) );
	ME_.set_afff                ( GetHexBits(MEc, 14, 14) );

	ME_.set_tbin_num            ( GetHexBits(MEd,  0,  2) );
	ME_.set_vp                  ( GetHexBits(MEd,  3,  3) );
	ME_.set_station             ( GetHexBits(MEd,  4,  6) );
	ME_.set_af                  ( GetHexBits(MEd,  7,  7) );
	ME_.set_epc                 ( GetHexBits(MEd,  8, 11) );
	ME_.set_sm                  ( GetHexBits(MEd, 12, 12) );
	ME_.set_se                  ( GetHexBits(MEd, 13, 13) );
	ME_.set_afef                ( GetHexBits(MEd, 14, 14) );

	// ME_.set_dataword            ( uint64_t dataword);

	int _endcap = ((res->at(iOut)).PtrEventHeader()->Endcap() == 1) ? 1 : 2;

	// Compute station, CSC ID, sector, subsector, and neighbor
	std::vector<int> conv_vals = convert_ME_location( ME_.Station(), ME_.CSC_ID(),
							  (res->at(iOut)).PtrEventHeader()->Sector() );
	int tmp_station   = conv_vals.at(0);
	int tmp_csc_ID    = conv_vals.at(1);
	int tmp_sector    = conv_vals.at(2);
	int tmp_subsector = conv_vals.at(3);
	int tmp_neighbor  = conv_vals.at(4);

	// Compute ring number
	int tmp_ring = -99;
	if (tmp_station > 1) {
	  if (tmp_csc_ID < 4) tmp_ring = 1;
	  else                tmp_ring = 2;
	}
	else {
	  if      (tmp_csc_ID < 4) tmp_ring = 1;
	  else if (tmp_csc_ID < 7) tmp_ring = 2;
	  else                     tmp_ring = 3;
	}

	// Compute chamber number
	int tmp_chamber = -99;
	if (tmp_station == 1) {
	  tmp_chamber = ((tmp_sector-1) * 6) + tmp_csc_ID + 2; // Chamber offset of 2: First chamber in sector 1 is chamber 3
	  if (tmp_ring == 2)       tmp_chamber -= 3;
	  if (tmp_ring == 3)       tmp_chamber -= 6;
	  if (tmp_subsector == 2)  tmp_chamber += 3;
	  if (tmp_chamber > 36)    tmp_chamber -= 36;
	}
	else if (tmp_ring == 1) {
	  tmp_chamber = ((tmp_sector-1) * 3) + tmp_csc_ID + 1; // Chamber offset of 1: First chamber in sector 1 is chamber 2
	  if (tmp_chamber > 18) tmp_chamber -= 18;
	}
	else if (tmp_ring == 2) {
	  tmp_chamber = ((tmp_sector-1) * 6) + tmp_csc_ID - 3 + 2; // Chamber offset of 2: First chamber in sector 1 is chamber 3
	  if (tmp_chamber > 36) tmp_chamber -= 36;
	}

	CSCDetId Id_ = CSCDetId( _endcap, tmp_station, tmp_ring, tmp_chamber );
	// Unsure of how to fill "trknmb" (first field) or "bx0" (before SE) - for now filling with 1 and 0. - AWB 05.05.16
	// mpclink = 0 (after Tbin_num) indicates unsorted.
	CSCCorrelatedLCTDigi LCT_ = CSCCorrelatedLCTDigi( 1, ME_.VP(), ME_.Quality(), ME_.Key_wire_group(), 
							  ME_.CLCT_key_half_strip(), ME_.CLCT_pattern(), ME_.LR(),
							  ME_.Tbin_num() + 3, 0, 0, ME_.SE(), tmp_csc_ID );

	// write: // Temporarily disable for DQM operation - AWB 09.04.16

	(res->at(iOut)).push_ME(ME_);
	if ( tmp_neighbor != 1 ) // Don't write duplicate LCTs from adjacent sectors
	  res_LCT->insertDigi( Id_, LCT_ );

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

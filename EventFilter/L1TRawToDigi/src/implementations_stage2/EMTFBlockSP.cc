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
	if(GetHexBits(SP1c, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1c are incorrect"; }
	if(GetHexBits(SP1d, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP1d are incorrect"; }
	if(GetHexBits(SP2a, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2a are incorrect"; }
	if(GetHexBits(SP2b, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2b are incorrect"; }
	if(GetHexBits(SP2c, 15, 15) != 1) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2c are incorrect"; }
	if(GetHexBits(SP2d, 15, 15) != 0) { errors += 1; edm::LogError("L1T|EMTF") << "Format identifier bits in SP2d are incorrect"; }

	return errors;

      }


      // Converts CSC_ID, sector, subsector, and neighbor                                                                                          
      std::vector<int> convert_SP_location(int _csc_ID, int _sector, int _subsector, int _station) {
        int new_sector = _sector;
        if (_station == 1) {
          if      (_csc_ID <  0) { int arr[] = {_csc_ID, -99, -99, -99}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID == 0) { int arr[] = { -1,  -1,  -1,  -1}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID <= 9) { int arr[] = {_csc_ID, new_sector, _subsector+1, 0}; std::vector<int> vec(arr, arr+4); return vec; }
          else new_sector = (_sector != 1) ? _sector-1 : 6;
	  
          if      (_csc_ID == 10) { int arr[] = {3, new_sector, 2, 1}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID == 11) { int arr[] = {6, new_sector, 2, 1}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID == 12) { int arr[] = {9, new_sector, 2, 1}; std::vector<int> vec(arr, arr+4); return vec; }
          else { int arr[] = {_csc_ID, -99, -99, -99}; std::vector<int> vec(arr, arr+4); return vec; }
        }
        else if (_station == 2 || _station == 3 || _station == 4) {
          if      (_csc_ID <  0) { int arr[] = {_csc_ID, -99, -99, -99}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID == 0) { int arr[] = { -1,  -1,  -1,  -1}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID <= 9) { int arr[] = {_csc_ID, new_sector, -1, 0}; std::vector<int> vec(arr, arr+4); return vec; }
          else new_sector = (_sector != 1) ? _sector-1 : 6;
	  
          if      (_csc_ID == 10) { int arr[] = {3, new_sector, -1, 1}; std::vector<int> vec(arr, arr+4); return vec; }
          else if (_csc_ID == 11) { int arr[] = {9, new_sector, -1, 1}; std::vector<int> vec(arr, arr+4); return vec; }
          else { int arr[] = {_csc_ID, -99, -99, -99}; std::vector<int> vec(arr, arr+4); return vec; }
        }
        else { int arr[] = {-99, -99, -99, -99}; std::vector<int> vec(arr, arr+4); return vec; }
      }

      bool SPBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// std::cout << "Inside EMTFBlockSP.cc: unpack" << std::endl;
	// LogDebug("L1T|EMTF") << "Inside EMTFBlockSP.cc: unpack"; // Why doesn't this work? - AWB 09.04.16
	
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

	// res is a pointer to a collection of EMTFDaqOut class objects
	// There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
	EMTFDaqOutCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
	int iOut = res->size() - 1;
	std::vector<int> conv_vals_SP;
	std::vector<int> conv_vals_pT_LUT;

	EMTFHitCollection* res_hit;
	res_hit = static_cast<EMTFCollections*>(coll)->getEMTFHits();

	EMTFTrackCollection* res_track;
        res_track = static_cast<EMTFCollections*>(coll)->getEMTFTracks();
        EMTFTrack Track_;

	RegionalMuonCandBxCollection* res_cand;
	res_cand = static_cast<EMTFCollections*>(coll)->getRegionalMuonCands();
	RegionalMuonCand mu_;
	
	// if (SP_.Format_Errors() > 0) goto write; // Temporarily disable for DQM operation - AWB 09.04.16

	///////////////////////////////////
	// Unpack the SP Output Data Record
	///////////////////////////////////

	SP_.set_phi_full     ( GetHexBits(SP1a,  0, 12) ); 
	SP_.set_c            ( GetHexBits(SP1a, 13, 13) );
	SP_.set_hl           ( GetHexBits(SP1a, 14, 14) );

	SP_.set_phi_GMT      ( TwosCompl(8, GetHexBits(SP1b, 0, 7)) );
	SP_.set_quality_GMT  ( GetHexBits(SP1b,  8, 11) );
	SP_.set_bc0          ( GetHexBits(SP1b, 12, 12) );
	SP_.set_se           ( GetHexBits(SP1b, 13, 13) );
	SP_.set_vc           ( GetHexBits(SP1b, 14, 14) );

	SP_.set_eta_GMT      ( TwosCompl(9, GetHexBits(SP1c, 0, 8)) );
	SP_.set_mode         ( GetHexBits(SP1c,  9, 12) );
	SP_.set_bx           ( GetHexBits(SP1c, 13, 14) );

	SP_.set_pt_GMT       ( GetHexBits(SP1d,  0,   8) );
	SP_.set_me1_stub_num ( GetHexBits(SP1d,  9,   9) );
	SP_.set_me1_CSC_ID   ( GetHexBits(SP1d, 10,  13) );
	SP_.set_me1_subsector( GetHexBits(SP1d, 14,  14) );

	SP_.set_me2_stub_num ( GetHexBits(SP2a,  0, 0 ) );
	SP_.set_me2_CSC_ID   ( GetHexBits(SP2a,  1, 4 ) );
	SP_.set_me3_stub_num ( GetHexBits(SP2a,  5, 5 ) );
	SP_.set_me3_CSC_ID   ( GetHexBits(SP2a,  6, 9 ) );
	SP_.set_me4_stub_num ( GetHexBits(SP2a, 10, 10) );
	SP_.set_me4_CSC_ID   ( GetHexBits(SP2a, 11, 14) );

	SP_.set_me1_delay    ( GetHexBits(SP2b,  0,  2) );
	SP_.set_me2_delay    ( GetHexBits(SP2b,  3,  5) );
	SP_.set_me3_delay    ( GetHexBits(SP2b,  6,  8) );
	SP_.set_me4_delay    ( GetHexBits(SP2b,  9, 11) );
	SP_.set_tbin         ( GetHexBits(SP2b, 12, 14) );

	SP_.set_pt_LUT_addr  ( GetHexBits(SP2c,  0, 14, SP2d,  0, 14) );

	// SP_.set_dataword     ( uint64_t dataword );

	Track_.ImportSP( SP_, (res->at(iOut)).PtrEventHeader()->Sector() );
	Track_.ImportPtLUT( Track_.Mode(), Track_.Pt_LUT_addr() );
	Track_.set_endcap       ( ((res->at(iOut)).PtrEventHeader()->Endcap() == 1) ? 1 : -1 );
        Track_.set_sector_index ( (Track_.Endcap() == 1) ? Track_.Sector() - 1 : Track_.Sector() + 5 );

	if ( (res->at(iOut)).PtrSPCollection()->size() > 0 )
	  if ( SP_.TBIN() == (res->at(iOut)).PtrSPCollection()->at( (res->at(iOut)).PtrSPCollection()->size() - 1 ).TBIN() )
	    Track_.set_track_num( (res->at(iOut)).PtrSPCollection()->size() );
	  else Track_.set_track_num( 0 );
	else Track_.set_track_num( 0 );
	
	mu_.setHwSign      ( SP_.C() );
	mu_.setHwSignValid ( SP_.VC() );
	mu_.setHwQual      ( SP_.Quality_GMT() );
	mu_.setHwEta       ( SP_.Eta_GMT() );
	mu_.setHwPhi       ( SP_.Phi_GMT() );
	mu_.setHwPt        ( SP_.Pt_GMT() );
	mu_.setTFIdentifiers ( Track_.Sector_GMT(), (Track_.Endcap() == 1) ? emtf_pos : emtf_neg );
	mu_.setTrackSubAddress( RegionalMuonCand::kTrkNum, Track_.Track_num() );
	// mu_.set_dataword   ( SP_.Dataword() );

	// set_type         ( _SP.() );
	// set_rank         ( _SP.() );
	// set_layer        ( _SP.() );
	// set_straightness ( _SP.() );
	// set_strip        ( _SP.() );
	// set_second_bx    ( _SP.() );
	// set_pt_XML       ( _SP.() );
	// set_theta_int    ( _SP.() );
	// set_isGMT        ( _SP.() );
	
	///////////////////////
	// Match hits to tracks
	///////////////////////

	int St_hits[4] = {0, 0, 0, 0}; // Number of matched hits in each station
	int dBX[3] = {0, 1, 2};        // Hit - track BX values for earliest LCT configuration
	// int dBX[3] = [0, 1, -1];      // Hit - track BX values for 2nd-earliest LCT configuration

	Track_.set_has_neighbor(false);
	Track_.set_all_neighbor(true);

	conv_vals_SP = convert_SP_location( SP_.ME1_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), SP_.ME1_subsector(), 1 );
	if ( conv_vals_SP.at(3) == 1 and not Track_.Has_neighbor() ) Track_.set_has_neighbor(true);
	if ( conv_vals_SP.at(3) == 0 and     Track_.All_neighbor() ) Track_.set_all_neighbor(false); 
	for (uint iBX = 0; iBX < 3; iBX++) { // Loop over BX values nearest to the track BX
	  for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	    if ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) && (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) &&
		 (res_hit->at(iHit)).Subsector() == conv_vals_SP.at(2) && (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) &&
		 (res_hit->at(iHit)).Station() == 1 && (res_hit->at(iHit)).Stub_num() == SP_.ME1_stub_num() &&
		 (res_hit->at(iHit)).BX() - (SP_.TBIN() - 3) == dBX[iBX] ) {
	      if (St_hits[0] == 0 ) Track_.push_Hit( res_hit->at(iHit) );
	      St_hits[0] += 1; }
	  }
	  if (St_hits[0] > 0) break; // If you found a hit in a BX close to the track, not need to look in other BX
	}
	mu_.setTrackSubAddress( RegionalMuonCand::kME1Seg, SP_.ME1_stub_num() );
	mu_.setTrackSubAddress( RegionalMuonCand::kME1Ch,  calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 1) );

	conv_vals_SP = convert_SP_location( SP_.ME2_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 2 );
	if ( conv_vals_SP.at(3) == 1 and not Track_.Has_neighbor() ) Track_.set_has_neighbor(true);
	if ( conv_vals_SP.at(3) == 0 and     Track_.All_neighbor() ) Track_.set_all_neighbor(false); 
	for (uint iBX = 0; iBX < 3; iBX++) { 
	  for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	    if ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) && (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) && 
	    // if ( ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) || (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) + 3 ) && 
	    // 	 (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) && 
		 (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) && (res_hit->at(iHit)).Station() == 2 && 
		 (res_hit->at(iHit)).Stub_num() == SP_.ME2_stub_num() && (res_hit->at(iHit)).BX() - (SP_.TBIN() - 3) == dBX[iBX] ) {
	      if (St_hits[1] == 0 ) Track_.push_Hit( res_hit->at(iHit) );
	      St_hits[1] += 1; }
	  }
	  if (St_hits[1] > 0) break; 
	}
	mu_.setTrackSubAddress( RegionalMuonCand::kME2Seg, SP_.ME2_stub_num() );
	mu_.setTrackSubAddress( RegionalMuonCand::kME2Ch,  calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 2) );

	conv_vals_SP = convert_SP_location( SP_.ME3_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 3 );
	if ( conv_vals_SP.at(3) == 1 and not Track_.Has_neighbor() ) Track_.set_has_neighbor(true);
	if ( conv_vals_SP.at(3) == 0 and     Track_.All_neighbor() ) Track_.set_all_neighbor(false); 
	for (uint iBX = 0; iBX < 3; iBX++) { 
	  for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	    if ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) && (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) &&
	    // if ( ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) || (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) + 3 ) && 
	    // 	 (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) && 
		 (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) && (res_hit->at(iHit)).Station() == 3 && 
		 (res_hit->at(iHit)).Stub_num() == SP_.ME3_stub_num() && (res_hit->at(iHit)).BX() - (SP_.TBIN() - 3) == dBX[iBX] ) {
	      if (St_hits[2] == 0 ) Track_.push_Hit( res_hit->at(iHit) );
	      St_hits[2] += 1; }
	  }
	  if (St_hits[2] > 0) break; 
	}
	mu_.setTrackSubAddress( RegionalMuonCand::kME3Seg, SP_.ME3_stub_num() );
	mu_.setTrackSubAddress( RegionalMuonCand::kME3Ch,  calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 3) );

	conv_vals_SP = convert_SP_location( SP_.ME4_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 4 );
	if ( conv_vals_SP.at(3) == 1 and not Track_.Has_neighbor() ) Track_.set_has_neighbor(true);
	if ( conv_vals_SP.at(3) == 0 and     Track_.All_neighbor() ) Track_.set_all_neighbor(false); 
	for (uint iBX = 0; iBX < 3; iBX++) { 
	  for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	    if ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) && (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) && 
	    // if ( ( (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) || (res_hit->at(iHit)).CSC_ID() == conv_vals_SP.at(0) + 3 ) && 
	    // 	 (res_hit->at(iHit)).Sector() == conv_vals_SP.at(1) && 
		 (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) && (res_hit->at(iHit)).Station() == 4 && 
		 (res_hit->at(iHit)).Stub_num() == SP_.ME4_stub_num() && (res_hit->at(iHit)).BX() - (SP_.TBIN() - 3) == dBX[iBX] ) {
	      if (St_hits[3] == 0 ) Track_.push_Hit( res_hit->at(iHit) );
	      St_hits[3] += 1; }
	  }
	  if (St_hits[3] > 0) break; 
	}
	mu_.setTrackSubAddress( RegionalMuonCand::kME4Seg, SP_.ME4_stub_num() );
	mu_.setTrackSubAddress( RegionalMuonCand::kME4Ch,  calc_uGMT_chamber(conv_vals_SP.at(0), conv_vals_SP.at(2), conv_vals_SP.at(3), 4) );


	// if ( Track_.Mode() != St_hits[0]*8 + St_hits[1]*4 + St_hits[2]*2 + St_hits[3] ) {
	//   std::cout << "" << std::endl;
	//   std::cout << "***********************************************************" << std::endl;
	//   std::cout << "Bug in EMTF event! Mode " << Track_.Mode() << " track with (" << St_hits[0] << ", " << St_hits[1] 
	// 	    << ", " << St_hits[2] << ", " << St_hits[3] << ") hits in stations (1, 2, 3, 4)" << std::endl;
	//   std::cout << "Sector = " << (res->at(iOut)).PtrEventHeader()->Sector() << ", ME1_stub_num = " << SP_.ME1_stub_num() << ", ME1_CSC_ID = " << SP_.ME1_CSC_ID() 
	// 	    << ", ME1_subsector = " << SP_.ME1_subsector() << std::endl;
	//   std::cout << "Sector = " << (res->at(iOut)).PtrEventHeader()->Sector() << ", ME2_stub_num = " << SP_.ME2_stub_num() << ", ME2_CSC_ID = " << SP_.ME2_CSC_ID() << std::endl;
	//   std::cout << "Sector = " << (res->at(iOut)).PtrEventHeader()->Sector() << ", ME3_stub_num = " << SP_.ME3_stub_num() << ", ME3_CSC_ID = " << SP_.ME3_CSC_ID() << std::endl;
	//   std::cout << "Sector = " << (res->at(iOut)).PtrEventHeader()->Sector() << ", ME4_stub_num = " << SP_.ME4_stub_num() << ", ME4_CSC_ID = " << SP_.ME4_CSC_ID() << std::endl;

	//   for (uint iHit = 0; iHit < res_hit->size(); iHit++)
	//     std::cout << "ID = " << (res_hit->at(iHit)).CSC_ID() << ", sector = " << (res_hit->at(iHit)).Sector()
	// 	      << ", sub = " << (res_hit->at(iHit)).Subsector() << ", neighbor = " << (res_hit->at(iHit)).Neighbor()
	// 	      << ", station = " << (res_hit->at(iHit)).Station() << ", stub = " << (res_hit->at(iHit)).Stub_num() 
	// 	      << ", BX = " << (res_hit->at(iHit)).BX() << ", ring = " << (res_hit->at(iHit)).Ring() 
	// 	      << ", chamber = " << (res_hit->at(iHit)).Chamber() << std::endl;

	//   for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	//     if (iHit == 0) (res_hit->at(iHit)).PrintSimulatorHeader();
	//     (res_hit->at(iHit)).PrintForSimulator();
	//   }
	//   std::cout << "***********************************************************" << std::endl;
	//   std::cout << "" << std::endl;
	// }

	// write: // Temporarily disable for DQM operation - AWB 09.04.16

	(res->at(iOut)).push_SP(SP_);

	res_track->push_back( Track_ );

	// TBIN_num can range from 0 through 7, i.e. BX = -3 through +4. - AWB 04.04.16
	res_cand->setBXRange(-3, 4);
	res_cand->push_back(SP_.TBIN() - 3, mu_);

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

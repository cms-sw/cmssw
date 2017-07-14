// Code to unpack the "SP Output Data Record"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

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

	// Check the number of 16-bit words
	if (payload.size() != 8) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Payload size in 'SP Output Data Record' is different than expected"; }

	// Check that each word is 16 bits
	for (unsigned int i = 0; i < 8; i++) {
	  if (GetHexBits(payload[i], 16, 31) != 0) { errors += 1; 
	    edm::LogError("L1T|EMTF") << "Payload[" << i << "] has more than 16 bits in 'SP Output Data Record'"; }
	}
	
	uint16_t SP1a = payload[0];
	uint16_t SP1b = payload[1];
	uint16_t SP1c = payload[2];
	uint16_t SP1d = payload[3];
	uint16_t SP2a = payload[4];
	uint16_t SP2b = payload[5];
	uint16_t SP2c = payload[6];
	uint16_t SP2d = payload[7];
      
	// Check Format
	if (GetHexBits(SP1a, 15, 15) != 1) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP1a are incorrect"; }
	if (GetHexBits(SP1b, 15, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP1b are incorrect"; }
	if (GetHexBits(SP1c, 15, 15) != 1) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP1c are incorrect"; }
	if (GetHexBits(SP1d, 15, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP1d are incorrect"; }
	if (GetHexBits(SP2a, 15, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP2a are incorrect"; }
	if (GetHexBits(SP2b, 15, 15) != 1) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP2b are incorrect"; }
	if (GetHexBits(SP2c, 15, 15) != 1) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP2c are incorrect"; }
	if (GetHexBits(SP2d, 15, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in SP2d are incorrect"; }

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

	ImportSP( Track_, SP_, (res->at(iOut)).PtrEventHeader()->Endcap(), (res->at(iOut)).PtrEventHeader()->Sector() );
	// Track_.ImportPtLUT( Track_.Mode(), Track_.Pt_LUT_addr() );  // Deprecated ... replace? - AWB 15.03.17

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
	mu_.setTFIdentifiers ( Track_.Sector() - 1, (Track_.Endcap() == 1) ? emtf_pos : emtf_neg );
	mu_.setTrackSubAddress( RegionalMuonCand::kTrkNum, Track_.Track_num() );
	// mu_.set_dataword   ( SP_.Dataword() );
	// Track_.set_GMT(mu_);

	///////////////////////
	// Match hits to tracks
	///////////////////////

	// Find the track delay
	int nDelay[3] = {0, 0, 0}; // Number of hits in the track with delay 0, 1, or 2
	if ( Track_.Mode()      >= 8) nDelay[SP_.ME1_delay()] += 1;
	if ((Track_.Mode() % 8) >= 4) nDelay[SP_.ME2_delay()] += 1;
	if ((Track_.Mode() % 4) >= 2) nDelay[SP_.ME3_delay()] += 1;
	if ((Track_.Mode() % 2) == 1) nDelay[SP_.ME4_delay()] += 1;

	int trk_delay = -99;
	// Assume 2nd-earliest LCT configuration
	if      (nDelay[2]                         >= 2) trk_delay = 2;
	else if (nDelay[2] + nDelay[1]             >= 2) trk_delay = 1;
	else if (nDelay[2] + nDelay[1] + nDelay[0] >= 2) trk_delay = 0;

	// // For earliest LCT configuration
	// if      (nDelay[2]                         >= 1) trk_delay = 2;
	// else if (nDelay[2] + nDelay[1]             >= 1) trk_delay = 1;
	// else if (nDelay[2] + nDelay[1] + nDelay[0] >= 1) trk_delay = 0;
	
	int St_hits[4] = {0, 0, 0, 0}; // Number of matched hits in each station

	for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	  
	  if ( (res_hit->at(iHit)).Endcap() != Track_.Endcap() ) continue;
	  
	  int hit_delay = -99;
	  if      ( (res_hit->at(iHit)).Station() == 1 ) hit_delay = SP_.ME1_delay();
	  else if ( (res_hit->at(iHit)).Station() == 2 ) hit_delay = SP_.ME2_delay();
	  else if ( (res_hit->at(iHit)).Station() == 3 ) hit_delay = SP_.ME3_delay();
	  else if ( (res_hit->at(iHit)).Station() == 4 ) hit_delay = SP_.ME4_delay();
	  
	  // Require exact matching according to TBIN and delays
	  if ( (res_hit->at(iHit)).BX() + 3 + hit_delay != SP_.TBIN() + trk_delay ) continue;
	  
	  // Match hit in station 1
	  conv_vals_SP = convert_SP_location( SP_.ME1_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), SP_.ME1_subsector(), 1 );
	  
	  if ( (res_hit->at(iHit)).Station()  == 1                  &&	    
	       (res_hit->at(iHit)).Sector()   == conv_vals_SP.at(1) &&
	       (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) &&
	       (res_hit->at(iHit)).Stub_num() == SP_.ME1_stub_num() ) {
	    
	    if ( (res_hit->at(iHit)).Is_CSC() == 1 && 
		 ( (res_hit->at(iHit)).CSC_ID()    != conv_vals_SP.at(0) ||
		   (res_hit->at(iHit)).Subsector() != conv_vals_SP.at(2) ) ) continue;
	    
	    int RPC_subsector = (((res_hit->at(iHit)).Subsector() - 1) / 3) + 1; // Map RPC subsector to equivalent CSC subsector
	    int RPC_CSC_ID    = (((res_hit->at(iHit)).Subsector() - 1) % 3) + 4; // Map RPC subsector and ring to equivalent CSC ID
	    
	    if ( (res_hit->at(iHit)).Is_RPC() == 1 &&
		 ( RPC_CSC_ID    != conv_vals_SP.at(0) ||
		   RPC_subsector != conv_vals_SP.at(2) ) ) continue;
	    
	    if (St_hits[0] == 0 ) { // Only add the first matched hit to the track
	      Track_.push_Hit( res_hit->at(iHit) );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME1Seg, SP_.ME1_stub_num() );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME1Ch,  
				      L1TMuonEndCap::calc_uGMT_chamber( conv_vals_SP.at(0), 
									conv_vals_SP.at(2), 
									conv_vals_SP.at(3), 1) );
	    }
	    St_hits[0] += 1; // Count the total number of matches for debugging purposes
	  } // End conditional: if ( (res_hit->at(iHit)).Station() == 1 
	  

	  // Match hit in station 2
	  conv_vals_SP = convert_SP_location( SP_.ME2_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 2 );
	  
	  if ( (res_hit->at(iHit)).Station()  == 2                  &&	    
	       (res_hit->at(iHit)).Sector()   == conv_vals_SP.at(1) &&
	       (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) &&
	       (res_hit->at(iHit)).Stub_num() == SP_.ME2_stub_num() ) {
	    
	    if ( (res_hit->at(iHit)).Is_CSC() == 1 && 
		 (res_hit->at(iHit)).CSC_ID() != conv_vals_SP.at(0) ) continue;
	    
	    if ( (res_hit->at(iHit)).Is_RPC() == 1 &&
		 (res_hit->at(iHit)).Subsector() + 3 != conv_vals_SP.at(0) ) continue;
	    
	    if (St_hits[1] == 0 ) {
	      Track_.push_Hit( res_hit->at(iHit) );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME2Seg, SP_.ME2_stub_num() );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME2Ch,  
				      L1TMuonEndCap::calc_uGMT_chamber( conv_vals_SP.at(0), 
									conv_vals_SP.at(2), 
									conv_vals_SP.at(3), 2) );
	    }
	    St_hits[1] += 1;
	  } // End conditional: if ( (res_hit->at(iHit)).Station() == 2 
	  

	  // Match hit in station 3
	  conv_vals_SP = convert_SP_location( SP_.ME3_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 3 );
	  
	  if ( (res_hit->at(iHit)).Station()  == 3                  &&	    
	       (res_hit->at(iHit)).Sector()   == conv_vals_SP.at(1) &&
	       (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) &&
	       (res_hit->at(iHit)).Stub_num() == SP_.ME3_stub_num() ) {
	    
	    if ( (res_hit->at(iHit)).Is_CSC() == 1 && 
		 (res_hit->at(iHit)).CSC_ID() != conv_vals_SP.at(0) ) continue;

	    if ( (res_hit->at(iHit)).Is_RPC() == 1 &&
		 (res_hit->at(iHit)).Subsector() + 3 != conv_vals_SP.at(0) ) continue;
	    
	    if (St_hits[2] == 0 ) {
	      Track_.push_Hit( res_hit->at(iHit) );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME3Seg, SP_.ME3_stub_num() );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME3Ch,  
				      L1TMuonEndCap::calc_uGMT_chamber( conv_vals_SP.at(0), 
									conv_vals_SP.at(2), 
									conv_vals_SP.at(3), 3) );
	    }
	    St_hits[2] += 1;
	  } // End conditional: if ( (res_hit->at(iHit)).Station() == 3 
	  
	  
	  // Match hit in station 4
	  conv_vals_SP = convert_SP_location( SP_.ME4_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 4 );
	  
	  if ( (res_hit->at(iHit)).Station()  == 4                  &&	    
	       (res_hit->at(iHit)).Sector()   == conv_vals_SP.at(1) &&
	       (res_hit->at(iHit)).Neighbor() == conv_vals_SP.at(3) &&
	       (res_hit->at(iHit)).Stub_num() == SP_.ME4_stub_num() ) {
	    
	    if ( (res_hit->at(iHit)).Is_CSC() == 1 && 
		 (res_hit->at(iHit)).CSC_ID() != conv_vals_SP.at(0) ) continue;
	    
	    if ( (res_hit->at(iHit)).Is_RPC() == 1 &&
		 (res_hit->at(iHit)).Subsector() + 3 != conv_vals_SP.at(0) ) continue;
	    
	    if (St_hits[3] == 0 ) {
	      Track_.push_Hit( res_hit->at(iHit) );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME4Seg, SP_.ME4_stub_num() );
	      mu_.setTrackSubAddress( RegionalMuonCand::kME4Ch,  
				      L1TMuonEndCap::calc_uGMT_chamber( conv_vals_SP.at(0), 
									conv_vals_SP.at(2), 
									conv_vals_SP.at(3), 4) );
	    }
	    St_hits[3] += 1;
	  } // End conditional: if ( (res_hit->at(iHit)).Station() == 4 
	  
	} // End loop: for (uint iHit = 0; iHit < res_hit->size(); iHit++)
      
	
	// if ( Track_.Mode() != St_hits[0]*8 + St_hits[1]*4 + St_hits[2]*2 + St_hits[3] && Track_.BX() == 0) {
	//   std::cout << "\n\n***********************************************************" << std::endl;
	//   std::cout << "Bug in unpacked EMTF event! Mode " << Track_.Mode() << " track in sector " << Track_.Sector()*Track_.Endcap() 
	// 	    << ", BX " << Track_.BX() << " (delay = " << trk_delay << ") with (" << St_hits[0] << ", " << St_hits[1] 
	// 	    << ", " << St_hits[2] << ", " << St_hits[3] << ") hits in stations (1, 2, 3, 4)" << std::endl;

	//   std::cout << "\nME1_stub_num = " << SP_.ME1_stub_num() << ", ME1_delay = " <<  SP_.ME1_delay() 
	// 	    << ", ME1_CSC_ID = " << SP_.ME1_CSC_ID() <<  ", ME1_subsector = " << SP_.ME1_subsector() << std::endl;
	//   std::cout << "ME2_stub_num = " << SP_.ME2_stub_num() << ", ME2_delay = " <<  SP_.ME2_delay() 
	// 	    << ", ME2_CSC_ID = " << SP_.ME2_CSC_ID() << std::endl;
	//   std::cout << "ME3_stub_num = " << SP_.ME3_stub_num() << ", ME3_delay = " <<  SP_.ME3_delay() 
	// 	    << ", ME3_CSC_ID = " << SP_.ME3_CSC_ID() << std::endl;
	//   std::cout << "ME4_stub_num = " << SP_.ME4_stub_num() << ", ME4_delay = " <<  SP_.ME4_delay() 
	// 	    << ", ME4_CSC_ID = " << SP_.ME4_CSC_ID() << std::endl;

	//   conv_vals_SP = convert_SP_location( SP_.ME1_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), SP_.ME1_subsector(), 1 );
	//   std::cout << "\nConverted ME1 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
	// 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << std::endl;
	//   conv_vals_SP = convert_SP_location( SP_.ME2_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 2 );
	//   std::cout << "Converted ME2 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
	// 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << std::endl;
	//   conv_vals_SP = convert_SP_location( SP_.ME3_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 3 );
	//   std::cout << "Converted ME3 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
	// 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << std::endl;
	//   conv_vals_SP = convert_SP_location( SP_.ME4_CSC_ID(), (res->at(iOut)).PtrEventHeader()->Sector(), -99, 4 );
	//   std::cout << "Converted ME4 CSC ID = " << conv_vals_SP.at(0) << ", sector = " << conv_vals_SP.at(1)
	// 	    << ", subsector = " << conv_vals_SP.at(2) << ", neighbor = " << conv_vals_SP.at(3) << "\n" << std::endl;

	  
	//   for (uint iHit = 0; iHit < res_hit->size(); iHit++)
	//     std::cout << "Hit: Is CSC = " << (res_hit->at(iHit)).Is_CSC() << ", CSC ID = " << (res_hit->at(iHit)).CSC_ID() 
	// 	      << ", sector = " << (res_hit->at(iHit)).Sector() << ", sub = " << (res_hit->at(iHit)).Subsector()
	// 	      << ", neighbor = " << (res_hit->at(iHit)).Neighbor() << ", station = " << (res_hit->at(iHit)).Station()
	// 	      << ", ring = " << (res_hit->at(iHit)).Ring() << ", chamber = " << (res_hit->at(iHit)).Chamber()
	// 	      << ", stub = " << (res_hit->at(iHit)).Stub_num() << ", BX = " << (res_hit->at(iHit)).BX() << std::endl;
	  
	//   // for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	//   //   if (iHit == 0) (res_hit->at(iHit)).PrintSimulatorHeader();
	//   //   (res_hit->at(iHit)).PrintForSimulator();
	//   // }
	//   std::cout << "***********************************************************" << std::endl;
	//   std::cout << "" << std::endl;
	// }
	
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

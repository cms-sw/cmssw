// Code to unpack the "RPC Data Record"

#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFCollections.h"
#include "EMTFUnpackerTools.h"

// This is the "header" - no EMTFBlockRPC.h file is needed
namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      class RPCBlockUnpacker : public Unpacker { // "RPCBlockUnpacker" inherits from "Unpacker"
      public:
	virtual int  checkFormat(const Block& block); 
	// virtual bool checkFormat() override; // Return "false" if block format does not match expected format
	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
	// virtual bool packBlock(const Block& block, UnpackerCollections *coll) override;
      };
      
      // class RPCBlockPacker : public Packer { // "RPCBlockPacker" inherits from "Packer"
      // public:
      // 	virtual bool unpack(const Block& block, UnpackerCollections *coll) override; // Apparently it's always good to use override in C++
      // };
      
    }
  }
}

namespace l1t {
  namespace stage2 {
    namespace emtf {
      
      int RPCBlockUnpacker::checkFormat(const Block& block) { 				
	
	auto payload = block.payload();
	int errors = 0;
	
	// Check the number of 16-bit words
	if (payload.size() != 4) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Payload size in 'RPC Data Record' is different than expected"; }
	
	// Check that each word is 16 bits
	for (unsigned int i = 0; i < 4; i++) {
	  if (GetHexBits(payload[i], 16, 31) != 0) { errors += 1; 
	    edm::LogError("L1T|EMTF") << "Payload[" << i << "] has more than 16 bits in 'RPC Data Record'"; }
	}
	
	uint16_t RPCa = payload[0];
	uint16_t RPCb = payload[1];
	uint16_t RPCc = payload[2];
	uint16_t RPCd = payload[3];
	
	// Check Format
	if (GetHexBits(RPCa, 11, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in RPCa are incorrect"; }
	if (GetHexBits(RPCb,  5,  7) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in RPCb are incorrect"; }
	if (GetHexBits(RPCb, 15, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in RPCb are incorrect"; }
	if (GetHexBits(RPCc, 12, 13) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in RPCc are incorrect"; }
	if (GetHexBits(RPCc, 15, 15) != 1) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in RPCc are incorrect"; }
	if (GetHexBits(RPCd,  4, 15) != 0) { errors += 1; 
	  edm::LogError("L1T|EMTF") << "Format identifier bits in RPCd are incorrect"; }

	return errors;
	
      }

      // Converts station, ring, sector, subsector, neighbor, and segment from the RPC output
      void convert_RPC_location(int& station, int& ring, int& sector, int& subsector, int& neighbor, int& segment,
				const int evt_sector, const int frame, const int word, const int link) {
	station   = -99;
	ring      = -99;
	sector    = -99;
	subsector = -99;
	neighbor  = -99;
	segment   = -99;

	// "link" is the "link index" field (0 - 6) in the EMTF DAQ document, not "link number" (1 - 7)
	// Neighbor indicated by link == 0
	sector    = (link != 0 ? evt_sector : (evt_sector == 1 ? 6 : evt_sector - 1) );
	subsector = (link != 0 ? link : 6);
	neighbor  = (link == 0 ? 1 : 0);
	segment   = (word % 2);

	if        (frame == 0) {
	  station = (word < 2 ? 1 : 2);
	  ring    = 2;
	} else if (frame == 1) {
	  station = 3;
	  ring    = (word < 2 ? 2 : 3);
	} else if (frame == 2) {
	  station = 4;
	  ring    = (word < 2 ? 2 : 3);
	}
      } // End function: void convert_RPC_location()

      bool RPCBlockUnpacker::unpack(const Block& block, UnpackerCollections *coll) {
	
	// std::cout << "Inside EMTFBlockRPC.cc: unpack" << std::endl;
	
	// Get the payload for this block, made up of 16-bit words (0xffff)
	// Format defined in MTF7Payload::getBlock() in src/Block.cc
	// payload[0] = bits 0-15, payload[1] = 16-31, payload[3] = 32-47, etc.
	auto payload = block.payload();
	
	// Check Format of Payload
	l1t::emtf::RPC RPC_;	
	for (int err = 0; err < checkFormat(block); err++) RPC_.add_format_error();
	
	// Assign payload to 16-bit words
	uint16_t RPCa = payload[0];
	uint16_t RPCb = payload[1];
	uint16_t RPCc = payload[2];
	uint16_t RPCd = payload[3];
	
	// res is a pointer to a collection of EMTFDaqOut class objects
	// There is one EMTFDaqOut for each MTF7 (60 deg. sector) in the event
	EMTFDaqOutCollection* res;
	res = static_cast<EMTFCollections*>(coll)->getEMTFDaqOuts();
	int iOut = res->size() - 1;

        EMTFHitCollection* res_hit;
        res_hit = static_cast<EMTFCollections*>(coll)->getEMTFHits();
        EMTFHit Hit_;

	// Also unpack into RPC digis? - AWB 15.03.17

	////////////////////////////
	// Unpack the RPC Data Record
	////////////////////////////
	
	RPC_.set_phi     ( GetHexBits(RPCa,  0, 10) );

	RPC_.set_theta   ( GetHexBits(RPCb,  0,  4) );
	RPC_.set_word    ( GetHexBits(RPCb,  8,  9) );
	RPC_.set_frame   ( GetHexBits(RPCb, 10, 11) );
	RPC_.set_link    ( GetHexBits(RPCb, 12, 14) ); // Link index (0 - 6); link number runs 1 - 7
	
	RPC_.set_rpc_bxn ( GetHexBits(RPCc,  0, 11) );
	RPC_.set_bc0     ( GetHexBits(RPCc, 14, 14) );
	
	RPC_.set_tbin    ( GetHexBits(RPCd,  0,  2) );
	RPC_.set_vp      ( GetHexBits(RPCd,  3,  3) );
	
	// RPC_.set_dataword            ( uint64_t dataword);


	// Convert specially-encoded RPC quantities
	int _station, _ring, _sector, _subsector, _neighbor, _segment;
	convert_RPC_location( _station, _ring, _sector, _subsector, _neighbor, _segment,
			      (res->at(iOut)).PtrEventHeader()->Sector(), RPC_.Frame(), RPC_.Word(), RPC_.Link() );

	Hit_.set_station       ( _station   );
	Hit_.set_ring          ( _ring      );
	Hit_.set_sector        ( _sector    );
	Hit_.set_subsector     ( _subsector );
	Hit_.set_sector_RPC    ( _subsector < 5 ? _sector : (_sector % 6) + 1);  // Rotate by 20 deg to match RPC convention in CMSSW
	Hit_.set_subsector_RPC ( ((_subsector + 1) % 6) + 1 );  // Rotate by 2 to match RPC convention in CMSSW (RPCDetId.h) 
	Hit_.set_chamber       ( (Hit_.Sector_RPC() - 1)*6 + Hit_.Subsector_RPC() );
	Hit_.set_neighbor      ( _neighbor  );
	Hit_.set_pc_segment    ( _segment   );
	Hit_.set_fs_segment    ( _segment   );
	Hit_.set_bt_segment    ( _segment   );

	// Fill the EMTFHit
	ImportRPC( Hit_, RPC_, (res->at(iOut)).PtrEventHeader()->Endcap(), (res->at(iOut)).PtrEventHeader()->Sector() );

	// Set the stub number for this hit
	// Each chamber can send up to 2 stubs per BX
	// Also count stubs in corresponding CSC chamber; RPC hit counting is on top of LCT counting
	Hit_.set_stub_num(0);
	// // See if matching hit is already in event record (from neighboring sector) 
	// bool duplicate_hit_exists = false;
	for (uint iHit = 0; iHit < res_hit->size(); iHit++) {
	  
	  if ( Hit_.BX()      == res_hit->at(iHit).BX()      && 
	       Hit_.Endcap()  == res_hit->at(iHit).Endcap()  &&
	       Hit_.Station() == res_hit->at(iHit).Station() &&
	       Hit_.Chamber() == res_hit->at(iHit).Chamber() ) {

	    if ( (res_hit->at(iHit).Is_CSC() == 1 && res_hit->at(iHit).Ring() == 2) ||
		 (res_hit->at(iHit).Is_RPC() == 1) ) { // RPC rings 2 and 3 both map to CSC ring 2 

	      if ( Hit_.Neighbor() == res_hit->at(iHit).Neighbor() ) {
		Hit_.set_stub_num( Hit_.Stub_num() + 1);
	      } // else if ( res_hit->at(iHit).Is_RPC()   == 1               &&
	      // 		  res_hit->at(iHit).Ring()     == Hit_.Ring()     &&
	      // 		  res_hit->at(iHit).Theta_fp() == Hit_.Theta_fp() &&
	      // 		  res_hit->at(iHit).Phi_fp()   == Hit_.Phi_fp()   ) {
	      // 	duplicate_hit_exists = true;
	      // }
	    }
	  }
	} // End loop: for (uint iHit = 0; iHit < res_hit->size(); iHit++)

	(res->at(iOut)).push_RPC(RPC_);
	res_hit->push_back(Hit_);
	
	// Finished with unpacking one RPC Data Record
	return true;
	
      } // End bool RPCBlockUnpacker::unpack
      
      // bool RPCBlockPacker::pack(const Block& block, UnpackerCollections *coll) {
      // 	std::cout << "Inside RPCBlockPacker::pack" << std::endl;
      // 	return true;
      // } // End bool RPCBlockPacker::pack
      
    } // End namespace emtf
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_UNPACKER(l1t::stage2::emtf::RPCBlockUnpacker);
// DEFINE_L1T_PACKER(l1t::stage2::RPCBlockPacker);

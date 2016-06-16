
#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "EMTFCollections.h"
#include "EMTFTokens.h"

namespace l1t {
  namespace stage2 {
    class EMTFSetup : public PackingSetup {
    public:
      // Not sure what this function does - AWB 27.01.16
      virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override {
	return std::unique_ptr<PackerTokens>(new EMTFTokens(cfg, cc));
      };
      
      // Not sure what this block does, or if it's necessary - AWB 27.01.16
      virtual void fillDescription(edm::ParameterSetDescription& desc) override {
	desc.addOptional<edm::InputTag>("EMTFInputLabelAWB")->setComment("for stage2");
      };

      // Ignore Packer functionality for now - AWB 27.01.16
      virtual PackerMap getPackers(int fed, unsigned int fw) override {
	PackerMap res;
	
	if (fed == 1402) {
	  // Use amc_no and board id 1 for packing
	  res[{1, 1}] = {
	    // "RegionalMuonEMTFPacker" should be defined in RegionalMuonEMTFPacker.cc - AWB 11.01.15
	    PackerFactory::get()->make("stage2::RegionalMuonEMTFPacker"), 
	    // Should we even be doing a MuonPacker? = AWB 11.01.15
	    PackerFactory::get()->make("stage2::MuonPacker"),
	  };
	}
	
	return res;
      }; // End virtual PackerMap getPackers
      
      // Not sure what this function does - AWB 27.01.16
      virtual void registerProducts(edm::stream::EDProducerBase& prod) override {
	prod.produces<RegionalMuonCandBxCollection>();
	prod.produces<EMTFDaqOutCollection>();
	prod.produces<EMTFHitCollection>();
	prod.produces<EMTFTrackCollection>();
	prod.produces<CSCCorrelatedLCTDigiCollection>();
      };
      
      // Not sure what this function does - AWB 27.01.16
      virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
	return std::unique_ptr<UnpackerCollections>(new EMTFCollections(e));
      };
      
      virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override {
	// std::cout << "Inside EMTFSetup.cc: getUnpackers" << std::endl;

	// Presumably need some logic based on fed, amc, etc (c.f. CaloSetup.cc) - AWB 11.01.16
	UnpackerMap res;
	
	// "RegionalMuonEMTFPacker" should be defined in RegionalMuonEMTFPacker.cc - AWB 11.01.15

	auto emtf_headers_unp   = UnpackerFactory::get()->make("stage2::emtf::HeadersBlockUnpacker");  // Unpack "AMC data header" and "Event Record Header"
	auto emtf_counters_unp  = UnpackerFactory::get()->make("stage2::emtf::CountersBlockUnpacker"); // Unpack "Block of Counters"
	auto emtf_me_unp       = UnpackerFactory::get()->make("stage2::emtf::MEBlockUnpacker");      // Unpack "ME Data Record"
	auto emtf_rpc_unp      = UnpackerFactory::get()->make("stage2::emtf::RPCBlockUnpacker");     // // Unpack "RPC Data Record"
	auto emtf_sp_unp       = UnpackerFactory::get()->make("stage2::emtf::SPBlockUnpacker");      // Unpack "SP Output Data Record"
	auto emtf_trailers_unp  = UnpackerFactory::get()->make("stage2::emtf::TrailersBlockUnpacker"); // Unpack "Event Record Trailer"
	
	// Index of res is block->header().getID(), matching block_patterns_ in src/Block.cc
	res[511] = emtf_headers_unp;
	res[2]   = emtf_counters_unp;
	res[3]   = emtf_me_unp;
	res[4]   = emtf_rpc_unp;
	res[101] = emtf_sp_unp;
	res[255] = emtf_trailers_unp;
	
	return res;
      }; // End virtual UnpackerMap getUnpackers
    }; // End class EMTFSetup : public PackingSetup 
  } // End namespace stage2
} // End namespace l1t

DEFINE_L1T_PACKING_SETUP(l1t::stage2::EMTFSetup);

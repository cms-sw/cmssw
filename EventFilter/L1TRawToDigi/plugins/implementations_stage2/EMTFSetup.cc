#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EMTFSetup.h"

namespace l1t {
  namespace stage2 {
    std::unique_ptr<PackerTokens> EMTFSetup::registerConsumes(const edm::ParameterSet& cfg,
                                                              edm::ConsumesCollector& cc) {
      return std::unique_ptr<PackerTokens>(new EMTFTokens(cfg, cc));
    }

    // Not sure what this block does, or if it's necessary - AWB 27.01.16
    void EMTFSetup::fillDescription(edm::ParameterSetDescription& desc) {
      desc.addOptional<edm::InputTag>("EMTFInputLabelAWB")->setComment("for stage2");
    }

    PackerMap EMTFSetup::getPackers(int fed, unsigned int fw) {
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
    }

    void EMTFSetup::registerProducts(edm::ProducesCollector prod) {
      prod.produces<RegionalMuonCandBxCollection>();
      prod.produces<EMTFDaqOutCollection>();
      prod.produces<EMTFHitCollection>();
      prod.produces<EMTFTrackCollection>();
      prod.produces<CSCCorrelatedLCTDigiCollection>();
      prod.produces<CPPFDigiCollection>();
    }

    std::unique_ptr<UnpackerCollections> EMTFSetup::getCollections(edm::Event& e) {
      return std::unique_ptr<UnpackerCollections>(new EMTFCollections(e));
    }

    UnpackerMap EMTFSetup::getUnpackers(int fed, int board, int amc, unsigned int fw) {
      // std::cout << "Inside EMTFSetup.cc: getUnpackers" << std::endl;

      // Presumably need some logic based on fed, amc, etc (c.f. CaloSetup.cc) - AWB 11.01.16
      UnpackerMap res;

      // "RegionalMuonEMTFPacker" should be defined in RegionalMuonEMTFPacker.cc - AWB 11.01.15

      auto emtf_headers_unp = UnpackerFactory::get()->make(
          "stage2::emtf::HeadersBlockUnpacker");  // Unpack "AMC data header" and "Event Record Header"
      auto emtf_counters_unp =
          UnpackerFactory::get()->make("stage2::emtf::CountersBlockUnpacker");             // Unpack "Block of Counters"
      auto emtf_me_unp = UnpackerFactory::get()->make("stage2::emtf::MEBlockUnpacker");    // Unpack "ME Data Record"
      auto emtf_rpc_unp = UnpackerFactory::get()->make("stage2::emtf::RPCBlockUnpacker");  // Unpack "RPC Data Record"
      auto emtf_sp_unp =
          UnpackerFactory::get()->make("stage2::emtf::SPBlockUnpacker");  // Unpack "SP Output Data Record"
      auto emtf_trailers_unp =
          UnpackerFactory::get()->make("stage2::emtf::TrailersBlockUnpacker");  // Unpack "Event Record Trailer"

      emtf_me_unp->setAlgoVersion(
          fw);  // Currently only the CSC LCT unpacking needs the firmware version, can add others as needed - AWB 09.04.18

      // Index of res is block->header().getID(), matching block_patterns_ in src/Block.cc
      res[511] = emtf_headers_unp;
      res[2] = emtf_counters_unp;
      res[3] = emtf_me_unp;
      res[4] = emtf_rpc_unp;
      res[101] = emtf_sp_unp;
      res[255] = emtf_trailers_unp;

      return res;
    }  // End virtual UnpackerMap getUnpackers
  }    // End namespace stage2
}  // End namespace l1t

DEFINE_L1T_PACKING_SETUP(l1t::stage2::EMTFSetup);

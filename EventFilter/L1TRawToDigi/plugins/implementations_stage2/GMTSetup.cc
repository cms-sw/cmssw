#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/RegionalMuonGMTUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/RegionalMuonGMTPacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/MuonPacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/IntermediateMuonUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/MuonUnpacker.h"

#include "GMTSetup.h"

#include <array>
#include <string>

namespace l1t {
  namespace stage2 {
    std::unique_ptr<PackerTokens> GMTSetup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) {
      return std::unique_ptr<PackerTokens>(new GMTTokens(cfg, cc));
    }

    void GMTSetup::fillDescription(edm::ParameterSetDescription& desc) {
      desc.addOptional<edm::InputTag>("BMTFInputLabel")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("OMTFInputLabel")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("EMTFInputLabel")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("ImdInputLabelBMTF")
          ->setComment("uGMT intermediate muon from BMTF after first sorting stage");
      desc.addOptional<edm::InputTag>("ImdInputLabelEMTFNeg")
          ->setComment("uGMT intermediate muon from neg. EMTF side after first sorting stage");
      desc.addOptional<edm::InputTag>("ImdInputLabelEMTFPos")
          ->setComment("uGMT intermediate muon from pos. EMTF side after first sorting stage");
      desc.addOptional<edm::InputTag>("ImdInputLabelOMTFNeg")
          ->setComment("uGMT intermediate muon from neg. OMTF side after first sorting stage");
      desc.addOptional<edm::InputTag>("ImdInputLabelOMTFPos")
          ->setComment("uGMT intermediate muon from pos. OMTF side after first sorting stage");
    }

    PackerMap GMTSetup::getPackers(int fed, unsigned int fw) {
      PackerMap res;
      if (fed == 1402) {
        auto gmt_in_packer = static_pointer_cast<l1t::stage2::RegionalMuonGMTPacker>(
            PackerFactory::get()->make("stage2::RegionalMuonGMTPacker"));
        if (fw >= 0x6000000) {
          gmt_in_packer->setIsRun3();
        }
        auto gmt_out_packer =
            static_pointer_cast<l1t::stage2::GMTMuonPacker>(PackerFactory::get()->make("stage2::GMTMuonPacker"));
        gmt_out_packer->setFed(fed);
        gmt_out_packer->setFwVersion(fw);
        // Use amc_no and board id 1 for packing
        res[{1, 1}] = {
            gmt_in_packer,
            gmt_out_packer,
            PackerFactory::get()->make("stage2::IntermediateMuonPacker"),
        };
      }
      return res;
    }

    void GMTSetup::registerProducts(edm::ProducesCollector prod) {
      prod.produces<RegionalMuonCandBxCollection>("BMTF");
      prod.produces<RegionalMuonCandBxCollection>("OMTF");
      prod.produces<RegionalMuonCandBxCollection>("EMTF");
      prod.produces<MuonBxCollection>("Muon");
      for (int i = 1; i < 6; ++i) {
        prod.produces<MuonBxCollection>("MuonCopy" + std::to_string(i));
      }
      prod.produces<MuonBxCollection>("imdMuonsBMTF");
      prod.produces<MuonBxCollection>("imdMuonsEMTFNeg");
      prod.produces<MuonBxCollection>("imdMuonsEMTFPos");
      prod.produces<MuonBxCollection>("imdMuonsOMTFNeg");
      prod.produces<MuonBxCollection>("imdMuonsOMTFPos");
    }

    std::unique_ptr<UnpackerCollections> GMTSetup::getCollections(edm::Event& e) {
      return std::unique_ptr<UnpackerCollections>(new GMTCollections(e));
    }

    UnpackerMap GMTSetup::getUnpackers(int fed, int board, int amc, unsigned int fw) {
      UnpackerMap res;

      // MP7 input link numbers are represented by even numbers starting from 0 (iLink=link*2)
      // input muons on links 36-71
      auto gmt_in_unp = static_pointer_cast<l1t::stage2::RegionalMuonGMTUnpacker>(
          UnpackerFactory::get()->make("stage2::RegionalMuonGMTUnpacker"));
      if (fw >= 0x6000000) {
        gmt_in_unp->setIsRun3();
      }

      for (int iLink = 72; iLink < 144; iLink += 2) {
        res[iLink] = gmt_in_unp;
      }

      // MP7 output link numbers are represented by odd numbers (oLink=link*2+1)
      // internal muons on links 24-31
      auto gmt_imd_unp = static_pointer_cast<l1t::stage2::IntermediateMuonUnpacker>(
          UnpackerFactory::get()->make("stage2::IntermediateMuonUnpacker"));
      gmt_imd_unp->setAlgoVersion(fw);
      for (int oLink = 49; oLink < 65; oLink += 2)
        res[oLink] = gmt_imd_unp;

      // output muons on links 0-23 (6 copies on 4 links each)
      std::array<std::shared_ptr<l1t::stage2::MuonUnpacker>, 6> gmt_out_unps;
      int i = 0;
      for (auto gmt_out_unp : gmt_out_unps) {
        gmt_out_unp =
            static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
        gmt_out_unp->setAlgoVersion(fw);
        gmt_out_unp->setFedNumber(fed);
        gmt_out_unp->setMuonCopy(i);

        int oLinkMin = i * 8 + 1;
        for (int oLink = oLinkMin; oLink < oLinkMin + 8; oLink += 2)
          res[oLink] = gmt_out_unp;

        ++i;
      }

      return res;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GMTSetup);

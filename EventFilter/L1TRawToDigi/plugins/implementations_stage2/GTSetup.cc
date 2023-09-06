#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/MuonPacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/MuonUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/EGammaUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/EtSumUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/ZDCUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/JetUnpacker.h"
#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/TauUnpacker.h"

#include "GTSetup.h"

namespace l1t {
  namespace stage2 {
    std::unique_ptr<PackerTokens> GTSetup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) {
      return std::unique_ptr<PackerTokens>(new GTTokens(cfg, cc));
    }

    void GTSetup::fillDescription(edm::ParameterSetDescription& desc) {
      desc.addOptional<edm::InputTag>("GtInputTag")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("ExtInputTag")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("MuonInputTag")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("ShowerInputTag")->setComment("for Run3");
      desc.addOptional<edm::InputTag>("EGammaInputTag")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("JetInputTag")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("TauInputTag")->setComment("for stage2");
      desc.addOptional<edm::InputTag>("EtSumInputTag")->setComment("for stage2");
    }

    PackerMap GTSetup::getPackers(int fed, unsigned int fw) {
      PackerMap res;

      if ((fed == 1404) || (fed == 1405)) {
        // Use board id 1 for packing
        //fed id 1404 corresponds to the production crate, 1405 to the test crate
        auto gt_muon_packer =
            static_pointer_cast<l1t::stage2::GTMuonPacker>(PackerFactory::get()->make("stage2::GTMuonPacker"));
        gt_muon_packer->setFed(fed);
        gt_muon_packer->setFwVersion(fw);
        res[{1, 1}] = {gt_muon_packer,
                       PackerFactory::get()->make("stage2::GTEGammaPacker"),
                       PackerFactory::get()->make("stage2::GTEtSumPacker"),
                       PackerFactory::get()->make("stage2::GTJetPacker"),
                       PackerFactory::get()->make("stage2::GTTauPacker"),
                       PackerFactory::get()->make("stage2::GlobalAlgBlkPacker"),
                       PackerFactory::get()->make("stage2::GlobalExtBlkPacker")};
      }

      return res;
    }

    void GTSetup::registerProducts(edm::ProducesCollector prod) {
      prod.produces<MuonBxCollection>("Muon");
      prod.produces<MuonShowerBxCollection>("MuonShower");
      prod.produces<EGammaBxCollection>("EGamma");
      prod.produces<EtSumBxCollection>("EtSum");
      prod.produces<EtSumBxCollection>("ZDCSum"); // added addition EtSum collection for ZDC  unpacker 
      prod.produces<JetBxCollection>("Jet");
      prod.produces<TauBxCollection>("Tau");
      prod.produces<GlobalAlgBlkBxCollection>();
      prod.produces<GlobalExtBlkBxCollection>();
      for (int i = 2; i < 7; ++i) {  // Collections from boards 2-6
        prod.produces<MuonBxCollection>("Muon" + std::to_string(i));
        prod.produces<MuonShowerBxCollection>("MuonShower" + std::to_string(i));
        prod.produces<EGammaBxCollection>("EGamma" + std::to_string(i));
        prod.produces<EtSumBxCollection>("EtSum" + std::to_string(i));
        prod.produces<EtSumBxCollection>("ZDCSum" + std::to_string(i));
        prod.produces<JetBxCollection>("Jet" + std::to_string(i));
        prod.produces<TauBxCollection>("Tau" + std::to_string(i));
      }
    }

    std::unique_ptr<UnpackerCollections> GTSetup::getCollections(edm::Event& e) {
      return std::unique_ptr<UnpackerCollections>(new GTCollections(e));
    }

    UnpackerMap GTSetup::getUnpackers(int fed, int board, int amc, unsigned int fw) {
      auto muon_unp =
          static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
      auto egamma_unp =
          static_pointer_cast<l1t::stage2::EGammaUnpacker>(UnpackerFactory::get()->make("stage2::EGammaUnpacker"));
      auto etsum_unp =
          static_pointer_cast<l1t::stage2::EtSumUnpacker>(UnpackerFactory::get()->make("stage2::EtSumUnpacker"));
      auto zdc_unp =
          static_pointer_cast<l1t::stage2::ZDCUnpacker>(UnpackerFactory::get()->make("stage2::ZDCUnpacker"));
      auto jet_unp = static_pointer_cast<l1t::stage2::JetUnpacker>(UnpackerFactory::get()->make("stage2::JetUnpacker"));
      auto tau_unp = static_pointer_cast<l1t::stage2::TauUnpacker>(UnpackerFactory::get()->make("stage2::TauUnpacker"));

      if (fw >= 0x10f2) {
        etsum_unp = static_pointer_cast<l1t::stage2::EtSumUnpacker>(
            UnpackerFactory::get()->make("stage2::EtSumUnpacker_0x10010057"));
      }

      auto alg_unp = UnpackerFactory::get()->make("stage2::GlobalAlgBlkUnpacker");
      auto ext_unp = UnpackerFactory::get()->make("stage2::GlobalExtBlkUnpacker");

      muon_unp->setAlgoVersion(fw);
      muon_unp->setFedNumber(fed);

      muon_unp->setMuonCopy(amc - 1);
      egamma_unp->setEGammaCopy(amc - 1);
      etsum_unp->setEtSumCopy(amc - 1);
      zdc_unp->setZDCSumCopy(amc - 1);
      jet_unp->setJetCopy(amc - 1);
      tau_unp->setTauCopy(amc - 1);

      UnpackerMap res;

      if ((fed == 1404) || (fed == 1405)) {
        // From the rx buffers
        // fed id 1404 corresponds to the production crate, 1405 to the test crate
        res[0] = muon_unp;
        res[2] = muon_unp;
        res[4] = muon_unp;
        res[6] = muon_unp;
        res[8] = egamma_unp;
        res[10] = egamma_unp;
        res[12] = jet_unp;
        res[14] = jet_unp;
        res[16] = tau_unp;
        res[18] = tau_unp;
        res[20] = etsum_unp;
        res[22] = zdc_unp;

        if (amc == 1) {  // only unpack first uGT board for the external signal inputs (single copy)
          res[24] = ext_unp;
          //res[22] = empty link no data
          res[26] = ext_unp;
          res[28] = ext_unp;
          res[30] = ext_unp;
        }

        //From tx buffers
        res[33] = alg_unp;
        res[35] = alg_unp;
        res[37] = alg_unp;
        res[39] = alg_unp;
        res[41] = alg_unp;
        res[43] = alg_unp;
        res[45] = alg_unp;
        res[47] = alg_unp;
        res[49] = alg_unp;
      }

      return res;
    }
  }  // namespace stage2
}  // namespace l1t

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GTSetup);

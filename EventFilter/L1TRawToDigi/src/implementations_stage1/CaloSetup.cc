#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "CaloCollections.h"
#include "CaloTokens.h"

namespace l1t {
   namespace stage1 {
      class CaloSetup : public PackingSetup {
         public:
            virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override {
               return std::unique_ptr<PackerTokens>(new CaloTokens(cfg, cc));
            };

            virtual void fillDescription(edm::ParameterSetDescription& desc) override {
               desc.addOptional<edm::InputTag>("TauInputLabel")->setComment("for stage1");
               desc.addOptional<edm::InputTag>("IsoTauInputLabel")->setComment("for stage1");
               desc.addOptional<edm::InputTag>("HFBitCountsInputLabel")->setComment("for stage1");
               desc.addOptional<edm::InputTag>("HFRingSumsInputLabel")->setComment("for stage1");
               desc.addOptional<edm::InputTag>("RegionInputLabel")->setComment("for stage1");
               desc.addOptional<edm::InputTag>("EmCandInputLabel")->setComment("for stage1");
            };

            virtual PackerMap getPackers(int fed, unsigned int fw) override {
               PackerMap res;

               res[{1, 0x2300}] = {
                  PackerFactory::get()->make("stage1::IsoEGammaPacker"),
                  PackerFactory::get()->make("stage1::NonIsoEGammaPacker"),
                  PackerFactory::get()->make("stage1::CentralJetPacker"),
                  PackerFactory::get()->make("stage1::ForwardJetPacker"),
                  PackerFactory::get()->make("stage1::TauPacker"),
                  PackerFactory::get()->make("stage1::IsoTauPacker"),
                  PackerFactory::get()->make("stage1::EtSumPacker"),
                  PackerFactory::get()->make("stage1::MissEtPacker"),
                  PackerFactory::get()->make("stage1::CaloSpareHFPacker"),
                  PackerFactory::get()->make("stage1::MissHtPacker"),
                  PackerFactory::get()->make("stage1::RCTEmRegionPacker"),
               };

               return res;
            };

            virtual void registerProducts(edm::stream::EDProducerBase& prod) override {
               prod.produces<L1CaloEmCollection>();
               prod.produces<CaloSpareBxCollection>("HFBitCounts");
               prod.produces<CaloSpareBxCollection>("HFRingSums");
               prod.produces<L1CaloRegionCollection>();
               prod.produces<CaloTowerBxCollection>();
               prod.produces<EGammaBxCollection>();
               prod.produces<EtSumBxCollection>();
               prod.produces<JetBxCollection>();
               prod.produces<TauBxCollection>("rlxTaus");
               prod.produces<TauBxCollection>("isoTaus");
            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
               return std::unique_ptr<UnpackerCollections>(new CaloCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override {
               UnpackerMap res;

               // FIXME Hard-coded firmware version for first 74x MC campaigns.
               // Will account for differences in the AMC payload, MP7 payload,
               // and unpacker setup.
               if ((fw >> 24) == 0xff) {
                  auto cjet_unp = UnpackerFactory::get()->make("stage1::legacy::CentralJetUnpacker");
                  auto fjet_unp = UnpackerFactory::get()->make("stage1::legacy::ForwardJetUnpacker");

                  if (fed == 1352) {
                     if (board == 0x200D) {
                        auto iegamma_unp = UnpackerFactory::get()->make("stage1::legacy::IsoEGammaUnpacker");
                        auto niegamma_unp = UnpackerFactory::get()->make("stage1::legacy::NonIsoEGammaUnpacker");
                        auto tau_unp = UnpackerFactory::get()->make("stage1::legacy::TauUnpacker");
                        auto isotau_unp = UnpackerFactory::get()->make("stage1::legacy::IsoTauUnpacker");
                        auto etsum_unp = UnpackerFactory::get()->make("stage1::legacy::EtSumUnpacker");
                        auto ring_unp = UnpackerFactory::get()->make("stage1::legacy::HFRingUnpacker");

                        res[1] = iegamma_unp;
                        res[2] = niegamma_unp;
                        res[3] = cjet_unp;
                        res[4] = fjet_unp;
                        res[5] = tau_unp;
                        res[6] = etsum_unp;
                        res[7] = ring_unp;
                        res[8] = isotau_unp;
                     }
                  }
               } else {
                  auto cjet_unp_Left = UnpackerFactory::get()->make("stage1::CentralJetUnpackerLeft");
                  auto fjet_unp_Left = UnpackerFactory::get()->make("stage1::ForwardJetUnpackerLeft");
                  auto iegamma_unp_Left = UnpackerFactory::get()->make("stage1::IsoEGammaUnpackerLeft");
                  auto niegamma_unp_Left = UnpackerFactory::get()->make("stage1::NonIsoEGammaUnpackerLeft");
                  auto tau_unp_Left = UnpackerFactory::get()->make("stage1::TauUnpackerLeft");
                  auto isotau_unp_Left = UnpackerFactory::get()->make("stage1::IsoTauUnpackerLeft");
                  auto cjet_unp_Right = UnpackerFactory::get()->make("stage1::CentralJetUnpackerRight");
                  auto fjet_unp_Right = UnpackerFactory::get()->make("stage1::ForwardJetUnpackerRight");
                  auto iegamma_unp_Right = UnpackerFactory::get()->make("stage1::IsoEGammaUnpackerRight");
                  auto niegamma_unp_Right = UnpackerFactory::get()->make("stage1::NonIsoEGammaUnpackerRight");
                  auto tau_unp_Right = UnpackerFactory::get()->make("stage1::TauUnpackerRight");
                  auto isotau_unp_Right = UnpackerFactory::get()->make("stage1::IsoTauUnpackerRight");
                  auto etsum_unp = UnpackerFactory::get()->make("stage1::EtSumUnpacker");
                  auto missetsum_unp = UnpackerFactory::get()->make("stage1::MissEtUnpacker");
                  auto calospare_unp = UnpackerFactory::get()->make("stage1::CaloSpareHFUnpacker");
                  auto misshtsum_unp = UnpackerFactory::get()->make("stage1::MissHtUnpacker");

                  if (fed == 1352) {
                     auto rct_unp = UnpackerFactory::get()->make("stage1::RCTEmRegionUnpacker");

                     // 4109 == 0x100D
                     if(board == 0x2300){  
                        res[77] = cjet_unp_Left;
                        res[79] = cjet_unp_Right;  
                        res[81] = fjet_unp_Left;
                        res[83] = fjet_unp_Right;
                        res[85] = iegamma_unp_Left;
                        res[87] = iegamma_unp_Right;
                        res[89] = niegamma_unp_Left;
                        res[91] = niegamma_unp_Right;
                        res[93] = etsum_unp;
                        res[95] = missetsum_unp;
                        res[97] = calospare_unp;
                        res[99] = misshtsum_unp;
                        res[101] = tau_unp_Left;
                        res[103] = tau_unp_Right;
                        res[105] = isotau_unp_Left;
                        res[107] = isotau_unp_Right;

                        for (int m=0;m<36;m++) {
                           res[m*2] = rct_unp;
                        }
                     }
                  }
               }
               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage1::CaloSetup);

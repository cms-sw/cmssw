#include "FWCore/Framework/interface/one/EDProducerBase.h"

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

            virtual PackerMap getPackers(int fed, int fw) override {
               PackerMap res;

               res[{1, 0x100D}] = {
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
               res[{1, 0x100E}] = {
                  PackerFactory::get()->make("stage1::RCTEmRegionPacker"),      
               };

               return res;
            };

            virtual void registerProducts(edm::one::EDProducerBase& prod) override {
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

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, int fw) override {
               UnpackerMap res;

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
                  auto rctRegion_unp = UnpackerFactory::get()->make("stage1::RCTRegionUnpacker");
                  auto rctEm_unp = UnpackerFactory::get()->make("stage1::RCTEmUnpacker");
                  
                  if(board == 4109){  

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
                  }
                  for (int m=0;m<36;m++) {
                    if (board == 4109) {
                      res[m*2] = rctRegion_unp;
                    }
                    else if (board == 4110) {
                      res[m*2] = rctEm_unp;
                    }
                  }
               }
               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage1::CaloSetup);

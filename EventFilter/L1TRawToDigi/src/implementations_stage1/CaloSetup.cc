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

               // Use amc id 1 for packing
               res[1] = {
                  PackerFactory::get()->make("stage1::IsoEGammaPacker"),
                  PackerFactory::get()->make("stage1::NonIsoEGammaPacker"),
                  PackerFactory::get()->make("stage1::CentralJetPacker"),
                  PackerFactory::get()->make("stage1::ForwardJetPacker"),
                  PackerFactory::get()->make("stage1::TauPacker"),
                  PackerFactory::get()->make("stage1::IsoTauPacker"),
                  PackerFactory::get()->make("stage1::EtSumPacker"),
                  PackerFactory::get()->make("stage1::HFRingPacker"),
               };

               return res;
            };

            virtual void registerProducts(edm::one::EDProducerBase& prod) override {
               prod.produces<CaloSpareBxCollection>("HFBitCounts");
               prod.produces<CaloSpareBxCollection>("HFRingSums");
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

            virtual UnpackerMap getUnpackers(int fed, int amc, int fw) override {
               auto iegamma_unp = UnpackerFactory::get()->make("stage1::IsoEGammaUnpacker");
               auto niegamma_unp = UnpackerFactory::get()->make("stage1::NonIsoEGammaUnpacker");
               auto cjet_unp = UnpackerFactory::get()->make("stage1::CentralJetUnpacker");
               auto fjet_unp = UnpackerFactory::get()->make("stage1::ForwardJetUnpacker");
               auto tau_unp = UnpackerFactory::get()->make("stage1::TauUnpacker");
               auto isotau_unp = UnpackerFactory::get()->make("stage1::IsoTauUnpacker");
               auto etsum_unp = UnpackerFactory::get()->make("stage1::EtSumUnpacker");
               auto ring_unp = UnpackerFactory::get()->make("stage1::HFRingUnpacker");

               UnpackerMap res;
               res[1] = iegamma_unp;
               res[2] = niegamma_unp;
               res[3] = cjet_unp;
               res[4] = fjet_unp;
               res[5] = tau_unp;
               res[6] = etsum_unp;
               res[7] = ring_unp;
               res[8] = isotau_unp;

               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage1::CaloSetup);

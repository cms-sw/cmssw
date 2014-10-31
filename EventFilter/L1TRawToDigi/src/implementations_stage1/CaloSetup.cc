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

               if (fed == 1) {
                  // Use amc id 1 for packing
                  res[1] = {
                     /* PackerFactory::get()->make("stage1::CaloTowerPacker"), */
                     /* PackerFactory::get()->make("stage1::EGammaPacker"), */
                     /* PackerFactory::get()->make("stage1::EtSumPacker"), */
                     PackerFactory::get()->make("stage1::JetPacker"),
                     /* PackerFactory::get()->make("stage1::TauPacker") */
                  };
               }

               return res;
            };

            virtual void registerProducts(edm::one::EDProducerBase& prod) override {
               prod.produces<CaloTowerBxCollection>();
               prod.produces<EGammaBxCollection>();
               prod.produces<EtSumBxCollection>();
               prod.produces<JetBxCollection>();
               prod.produces<TauBxCollection>();
            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
               return std::unique_ptr<UnpackerCollections>(new CaloCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int amc, int fw) override {
               auto tower_unp = UnpackerFactory::get()->make("stage1::CaloTowerUnpacker");
               auto egamma_unp = UnpackerFactory::get()->make("stage1::EGammaUnpacker");
               auto etsum_unp = UnpackerFactory::get()->make("stage1::EtSumUnpacker");
               auto jet_unp = UnpackerFactory::get()->make("stage1::JetUnpacker");
               auto tau_unp = UnpackerFactory::get()->make("stage1::TauUnpacker");

               UnpackerMap res;
               if (fed == 1) {
                  /* res[1] = egamma_unp; */
                  /* res[3] = etsum_unp; */
                  res[3] = jet_unp;
                  /* res[7] = tau_unp; */
               }

               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage1::CaloSetup);

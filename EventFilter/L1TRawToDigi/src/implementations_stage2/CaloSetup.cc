#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "CaloCollections.h"
#include "CaloTokens.h"

namespace l1t {
   namespace stage2 {
      class CaloSetup : public PackingSetup {
         public:
            virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override {
               return std::unique_ptr<PackerTokens>(new CaloTokens(cfg, cc));
            };

            virtual void fillDescription(edm::ParameterSetDescription& desc) override {};

            virtual PackerMap getPackers(int fed, unsigned int fw) override {
               PackerMap res;

               if (fed == 1366) {
                  // Use board id 1 for packing
                  res[{1, 1}] = {
                     PackerFactory::get()->make("stage2::CaloTowerPacker"),
                     PackerFactory::get()->make("stage2::EGammaPacker"),
                     PackerFactory::get()->make("stage2::EtSumPacker"),
                     PackerFactory::get()->make("stage2::JetPacker"),
                     PackerFactory::get()->make("stage2::TauPacker")
                  };
               }

               return res;
            };

            virtual void registerProducts(edm::stream::EDProducerBase& prod) override {
               prod.produces<CaloTowerBxCollection>();
               prod.produces<EGammaBxCollection>();
               prod.produces<EtSumBxCollection>();
               prod.produces<JetBxCollection>();
               prod.produces<TauBxCollection>();

               prod.produces<EtSumBxCollection>("MP");
               prod.produces<JetBxCollection>("MP");
            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
               return std::unique_ptr<UnpackerCollections>(new CaloCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override {
               auto tower_unp = UnpackerFactory::get()->make("stage2::CaloTowerUnpacker");
               auto egamma_unp = UnpackerFactory::get()->make("stage2::EGammaUnpacker");
               auto etsum_unp = UnpackerFactory::get()->make("stage2::EtSumUnpacker");
               auto jet_unp = UnpackerFactory::get()->make("stage2::JetUnpacker");
               auto tau_unp = UnpackerFactory::get()->make("stage2::TauUnpacker");

               auto mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker");

               UnpackerMap res;
               if (fed == 1366) {
                  res[13] = jet_unp;
                  res[15] = jet_unp;
                  res[21] = etsum_unp;
               } else if (fed == 1360) {
                  res[1] = mp_unp;
                  res[3] = mp_unp;
                  res[5] = mp_unp;
                  res[7] = mp_unp;
                  res[9] = mp_unp;
                  res[11] = mp_unp;

                  for (int link = 0; link < 144; link += 2)
                     res[link] = tower_unp;
               }

               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::CaloSetup);

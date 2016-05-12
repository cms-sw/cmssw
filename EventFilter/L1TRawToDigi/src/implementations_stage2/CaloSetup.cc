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

            virtual void fillDescription(edm::ParameterSetDescription& desc) override {
	      desc.addOptional<edm::InputTag>("TowerInputLabel")->setComment("for stage 2");
	    };

            virtual PackerMap getPackers(int fed, unsigned int fw) override {
               PackerMap res;

               if (fed == 1366) {
                  // Use board id 1 for packing
                  res[{1, 1}] = {
		    //                     PackerFactory::get()->make("stage2::CaloTowerPacker"),
                     PackerFactory::get()->make("stage2::CaloEGammaPacker"),
                     PackerFactory::get()->make("stage2::CaloEtSumPacker"),
                     PackerFactory::get()->make("stage2::CaloJetPacker"),
                     PackerFactory::get()->make("stage2::CaloTauPacker")
                  };
               }

               return res;
            };

            virtual void registerProducts(edm::stream::EDProducerBase& prod) override {
               prod.produces<CaloTowerBxCollection>("CaloTower");
               prod.produces<EGammaBxCollection>("EGamma");
               prod.produces<EtSumBxCollection>("EtSum");
               prod.produces<JetBxCollection>("Jet");
               prod.produces<TauBxCollection>("Tau");

               prod.produces<EtSumBxCollection>("MP");
               prod.produces<JetBxCollection>("MP");
	       prod.produces<EGammaBxCollection>("MP");
	       prod.produces<TauBxCollection>("MP");
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
	       if (fw >= 0x1001000b) {
		 mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker_0x1001000b");
	       }
	       if (fw >= 0x10010010) {
		 mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker_0x10010010");
	       }
	       

               UnpackerMap res;
               if (fed == 1366 || (fed == 1360 && board == 0x221B)) {
	          res[9]  = egamma_unp;
	          res[11] = egamma_unp;
                  res[13] = jet_unp;
		  res[15] = jet_unp;
		  res[17] = tau_unp;
		  res[19] = tau_unp;
                  res[21] = etsum_unp;
	       } else if (fed == 1360 && board != 0x221B) {
                  res[121] = mp_unp;
                  res[123] = mp_unp;
                  res[125] = mp_unp;
                  res[127] = mp_unp;
                  res[129] = mp_unp;
                  res[131] = mp_unp;

                  for (int link = 0; link < 144; link += 2)
                     res[link] = tower_unp;
               }

               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::CaloSetup);

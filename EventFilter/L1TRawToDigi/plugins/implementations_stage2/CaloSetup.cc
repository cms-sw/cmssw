#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CaloSetup.h"

namespace l1t {
   namespace stage2 {
      std::unique_ptr<PackerTokens>
      CaloSetup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
      {
         return std::unique_ptr<PackerTokens>(new CaloTokens(cfg, cc));
      }

      void
      CaloSetup::fillDescription(edm::ParameterSetDescription& desc)
      {
         desc.addOptional<edm::InputTag>("TowerInputLabel")->setComment("for stage 2");
      }

      PackerMap
      CaloSetup::getPackers(int fed, unsigned int fw)
      {
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
      }

      void
      CaloSetup::registerProducts(edm::stream::EDProducerBase& prod)
      {
         prod.produces<CaloTowerBxCollection>("CaloTower");
         prod.produces<EGammaBxCollection>("EGamma");
         prod.produces<EtSumBxCollection>("EtSum");
         prod.produces<JetBxCollection>("Jet");
         prod.produces<TauBxCollection>("Tau");

         prod.produces<EtSumBxCollection>("MP");
         prod.produces<JetBxCollection>("MP");
         prod.produces<EGammaBxCollection>("MP");
         prod.produces<TauBxCollection>("MP");
      }

      std::unique_ptr<UnpackerCollections>
      CaloSetup::getCollections(edm::Event& e)
      {
         return std::unique_ptr<UnpackerCollections>(new CaloCollections(e));
      }

      UnpackerMap
      CaloSetup::getUnpackers(int fed, int board, int amc, unsigned int fw)
      {

	UnpackerMap res;
	if (fed == 1366 || (fed == 1360 && board == 0x221B)) {

	  auto egamma_unp = UnpackerFactory::get()->make("stage2::EGammaUnpacker");
	  auto etsum_unp = UnpackerFactory::get()->make("stage2::EtSumUnpacker");
	  auto jet_unp = UnpackerFactory::get()->make("stage2::JetUnpacker");
	  auto tau_unp = UnpackerFactory::get()->make("stage2::TauUnpacker");

	  if (fw >= 0x10010057) {
	    etsum_unp = UnpackerFactory::get()->make("stage2::EtSumUnpacker_0x10010057");
	  }

	  res[9]  = egamma_unp;
	  res[11] = egamma_unp;
	  res[13] = jet_unp;
	  res[15] = jet_unp;
	  res[17] = tau_unp;
	  res[19] = tau_unp;
	  res[21] = etsum_unp;

	} else if (fed == 1360 && board != 0x221B) {

	  auto tower_unp = UnpackerFactory::get()->make("stage2::CaloTowerUnpacker");
	  auto mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker");
	  if (fw >= 0x1001000b) {
            mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker_0x1001000b");
	  }
	  if (fw >= 0x10010010) {
            mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker_0x10010010");
	  }
	  if (fw >= 0x10010033) {
            mp_unp = UnpackerFactory::get()->make("stage2::MPUnpacker_0x10010033");
	  }

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
      }
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::CaloSetup);

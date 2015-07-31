#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "GTCollections.h"
#include "GTTokens.h"

namespace l1t {
   namespace stage2 {
      class GTSetup : public PackingSetup {
         public:
            virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override {
               return std::unique_ptr<PackerTokens>(new GTTokens(cfg, cc));
            };

            virtual void fillDescription(edm::ParameterSetDescription& desc) override {};

            virtual PackerMap getPackers(int fed, unsigned int fw) override {
               PackerMap res;

               if (fed == 1404) { 
                  // Use board id 1 for packing
                  res[{1, 1}] = {

		     PackerFactory::get()->make("stage2::EGammaPacker"),
		     PackerFactory::get()->make("stage2::EtSumPacker"),
		     PackerFactory::get()->make("stage2::JetPacker"),
		     PackerFactory::get()->make("stage2::TauPacker"),
                     PackerFactory::get()->make("stage2::GlobalAlgBlkPacker"),
                     PackerFactory::get()->make("stage2::GlobalExtBlkPacker")
                  };
               }

               return res;
            };

            virtual void registerProducts(edm::stream::EDProducerBase& prod) override {
	      
	       prod.produces<EGammaBxCollection>("GT");
	       prod.produces<EtSumBxCollection>("GT");
	       prod.produces<JetBxCollection>("GT");
	       prod.produces<TauBxCollection>("GT");
               prod.produces<GlobalAlgBlkBxCollection>();
               prod.produces<GlobalExtBlkBxCollection>();

            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
               return std::unique_ptr<UnpackerCollections>(new GTCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override {

  	       auto egamma_unp = UnpackerFactory::get()->make("stage2::EGammaUnpacker");
	       auto etsum_unp = UnpackerFactory::get()->make("stage2::EtSumUnpacker");
	       auto jet_unp = UnpackerFactory::get()->make("stage2::JetUnpacker");
	       auto tau_unp = UnpackerFactory::get()->make("stage2::TauUnpacker");
               auto alg_unp = UnpackerFactory::get()->make("stage2::GlobalAlgBlkUnpacker");
               auto ext_unp = UnpackerFactory::get()->make("stage2::GlobalExtBlkUnpacker");


               UnpackerMap res;
	       
               if (fed == 1404) {
                  
		 // Need to fill other input collections         
		  res[12] = jet_unp;
                  res[14] = jet_unp;
                  res[20] = etsum_unp;

                  res[1] = alg_unp;
                  res[3] = alg_unp;
                  res[5] = alg_unp;
                  
               }
	       
               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GTSetup);

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
                     
		     PackerFactory::get()->make("stage2::MuonPacker"),
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
	      
	       prod.produces<MuonBxCollection>("GT");
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

               auto muon_unp = UnpackerFactory::get()->make("stage2::MuonUnpacker");
  	       auto egamma_unp = UnpackerFactory::get()->make("stage2::EGammaUnpacker");
	       auto etsum_unp = UnpackerFactory::get()->make("stage2::EtSumUnpacker");
	       auto jet_unp = UnpackerFactory::get()->make("stage2::JetUnpacker");
	       auto tau_unp = UnpackerFactory::get()->make("stage2::TauUnpacker");
               auto alg_unp = UnpackerFactory::get()->make("stage2::GlobalAlgBlkUnpacker");
               auto ext_unp = UnpackerFactory::get()->make("stage2::GlobalExtBlkUnpacker");


               UnpackerMap res;
	       
               if (fed == 1404) {
                  
		 // From the rx buffers         
		  res[0]  = muon_unp;
		  res[2]  = muon_unp;
		  res[4]  = muon_unp;
		  res[6]  = muon_unp;
		  res[8]  = egamma_unp;
		  res[10] = egamma_unp;
		  res[12] = jet_unp;
                  res[14] = jet_unp;
		  res[16] = tau_unp;
		  res[18] = tau_unp;
                  res[20] = etsum_unp;
		  res[24] = ext_unp;
		  //res[22] = empty link no data
		  res[26] = ext_unp;
		  res[28] = ext_unp;
		  
                  //From the tx buffers
                  res[1]  = alg_unp;
                  res[3]  = alg_unp;
                  res[5]  = alg_unp;
                  res[7]  = alg_unp;
                  res[9]  = alg_unp;
                  res[11] = alg_unp;
                  res[13] = alg_unp;
                  res[15] = alg_unp;
                  res[17] = alg_unp;		  

                  
               }
	       
               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GTSetup);

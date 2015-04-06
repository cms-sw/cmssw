#include "FWCore/Framework/interface/one/EDProducerBase.h"

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

            virtual PackerMap getPackers(int fed, int fw) override {
               PackerMap res;

               if (fed == 1404) { 
                  // Use board id 1 for packing
                  res[{1, 1}] = {

                     PackerFactory::get()->make("stage2::GlobalAlgBlkPacker"),
                     PackerFactory::get()->make("stage2::GlobalExtBlkPacker")
                  };
               }

               return res;
            };

            virtual void registerProducts(edm::one::EDProducerBase& prod) override {

               prod.produces<GlobalAlgBlkBxCollection>();
               prod.produces<GlobalExtBlkBxCollection>();

            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
               return std::unique_ptr<UnpackerCollections>(new GTCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, int fw) override {

               auto alg_unp = UnpackerFactory::get()->make("stage2::GlobalAlgBlkUnpacker");
               auto ext_unp = UnpackerFactory::get()->make("stage2::GlobalExtBlkUnpacker");


               UnpackerMap res;
	       
               if (fed == 1404) {
                  
		 // Need to fill input collections         
                  
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

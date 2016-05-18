#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "BMTFCollections.h"
#include "BMTFTokens.h"

namespace l1t {
   namespace stage2 {
      class BMTFSetup : public PackingSetup {
         public:
            virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override {
               return std::unique_ptr<PackerTokens>(new BMTFTokens(cfg, cc));
            };

            virtual void fillDescription(edm::ParameterSetDescription& desc) override {};

            virtual PackerMap getPackers(int fed, unsigned int fw) override 
            {
               PackerMap res;

/*               if (fed == 1360) {
                  // Use board id 1 for packing
                  res[{1, 1}] = {
                     PackerFactory::get()->make("stage2::MuonPacker")
                  };
               }
*/
               return res;
            };

            virtual void registerProducts(edm::stream::EDProducerBase& prod) override 
            {
               prod.produces<RegionalMuonCandBxCollection>("BMTF");
               //prod.produces<L1MuDTChambPhContainer>("PhiDigis");
               //prod.produces<L1MuDTChambThContainer>("TheDigis");
               prod.produces<L1MuDTChambPhContainer>();
               prod.produces<L1MuDTChambThContainer>();
            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override 
            {
               return std::unique_ptr<UnpackerCollections>(new BMTFCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override 
            {
               auto outputMuon = UnpackerFactory::get()->make("stage2::BMTFUnpackerOutput");
               auto inputMuons = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputs");

               UnpackerMap res;
               if (fed == 1376 || fed == 1377 )
               {
									
   					for(int iL = 0; iL <= 70; iL += 2)
   					{
   						if ( iL == 12 || iL == 14 || ( iL > 26 && iL < 32) || iL == 60 || iL == 62 )
   							continue;
   						
   						res[iL] = inputMuons;
   					}
   					
   					res[123] = outputMuon;
					}
               
               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::BMTFSetup);

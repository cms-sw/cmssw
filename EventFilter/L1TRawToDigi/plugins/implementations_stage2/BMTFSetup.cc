#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "BMTFSetup.h"

namespace l1t {
   namespace stage2 {
      std::unique_ptr<PackerTokens>
      BMTFSetup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
      {
         return std::unique_ptr<PackerTokens>(new BMTFTokens(cfg, cc));
      }

      void BMTFSetup::fillDescription(edm::ParameterSetDescription& desc) {};

      PackerMap
      BMTFSetup::getPackers(int fed, unsigned int fw)
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
      }

      void
      BMTFSetup::registerProducts(edm::stream::EDProducerBase& prod)
      {
         prod.produces<RegionalMuonCandBxCollection>("BMTF");
         //prod.produces<L1MuDTChambPhContainer>("PhiDigis");
         //prod.produces<L1MuDTChambThContainer>("TheDigis");
         prod.produces<L1MuDTChambPhContainer>();
         prod.produces<L1MuDTChambThContainer>();
      }

      std::unique_ptr<UnpackerCollections>
      BMTFSetup::getCollections(edm::Event& e)
      {
         return std::unique_ptr<UnpackerCollections>(new BMTFCollections(e));
      }

      UnpackerMap
      BMTFSetup::getUnpackers(int fed, int board, int amc, unsigned int fw)
      {
         auto outputMuon = UnpackerFactory::get()->make("stage2::BMTFUnpackerOutput");
         auto inputMuonsOld = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputsOldQual");
         auto inputMuonsNew = UnpackerFactory::get()->make("stage2::BMTFUnpackerInputsNewQual");

         UnpackerMap res;
         if (fed == 1376 || fed == 1377) {
            for (int iL = 0; iL <= 70; iL += 2) {
               if (iL == 12 || iL == 14 || ( iL > 26 && iL < 32) || iL == 60 || iL == 62)
                  continue;

               if (fw < 2452619552)
                  res[iL] = inputMuonsOld;
               else
                  res[iL] = inputMuonsNew;
            }
            res[123] = outputMuon;
         }
         return res;
      };
   };
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::BMTFSetup);

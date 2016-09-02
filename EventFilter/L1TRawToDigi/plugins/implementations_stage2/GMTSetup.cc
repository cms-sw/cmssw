#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "GMTSetup.h"

namespace l1t {
   namespace stage2 {
      std::unique_ptr<PackerTokens>
      GMTSetup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
      {
         return std::unique_ptr<PackerTokens>(new GMTTokens(cfg, cc));
      }

      void
      GMTSetup::fillDescription(edm::ParameterSetDescription& desc)
      {
         desc.addOptional<edm::InputTag>("BMTFInputLabel")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("OMTFInputLabel")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("EMTFInputLabel")->setComment("for stage2");
      }

      PackerMap
      GMTSetup::getPackers(int fed, unsigned int fw)
      {
         PackerMap res;

         if (fed == 1402) {
            // Use amc_no and board id 1 for packing
            res[{1, 1}] = {
               PackerFactory::get()->make("stage2::RegionalMuonGMTPacker"),
               PackerFactory::get()->make("stage2::GMTMuonPacker"),
            };
         }

         return res;
      }

      void
      GMTSetup::registerProducts(edm::stream::EDProducerBase& prod)
      {
         prod.produces<RegionalMuonCandBxCollection>("BMTF");
         prod.produces<RegionalMuonCandBxCollection>("OMTF");
         prod.produces<RegionalMuonCandBxCollection>("EMTF");
         prod.produces<MuonBxCollection>("Muon");
      }

      std::unique_ptr<UnpackerCollections>
      GMTSetup::getCollections(edm::Event& e)
      {
         return std::unique_ptr<UnpackerCollections>(new GMTCollections(e));
      }

      UnpackerMap
      GMTSetup::getUnpackers(int fed, int board, int amc, unsigned int fw)
      {
         UnpackerMap res;

         auto gmt_in_unp = UnpackerFactory::get()->make("stage2::RegionalMuonGMTUnpacker");
         auto gmt_out_unp = UnpackerFactory::get()->make("stage2::MuonUnpacker");

         // input muons
         for (int iLink = 72; iLink < 144; iLink += 2)
            res[iLink] = gmt_in_unp;
         // output muons
         for (int oLink = 1; oLink < 9; oLink += 2)
            res[oLink] = gmt_out_unp;
         // internal muons
         //for (int oLink = 9; oLink < 24; oLink += 2)
         //    res[oLink] = gmt_out_unp;

         return res;
      }
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GMTSetup);

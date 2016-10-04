#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/Packer.h"
#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

#include "GMTCollections.h"
#include "GMTTokens.h"

namespace l1t {
   namespace stage2 {
      class GMTSetup : public PackingSetup {
         public:
            virtual std::unique_ptr<PackerTokens> registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc) override {
               return std::unique_ptr<PackerTokens>(new GMTTokens(cfg, cc));
            };

            virtual void fillDescription(edm::ParameterSetDescription& desc) override {
               desc.addOptional<edm::InputTag>("BMTFInputLabel")->setComment("for stage2");
               desc.addOptional<edm::InputTag>("OMTFInputLabel")->setComment("for stage2");
               desc.addOptional<edm::InputTag>("EMTFInputLabel")->setComment("for stage2");
               desc.addOptional<edm::InputTag>("ImdInputLabelBMTF")->setComment("uGMT intermediate muon from BMTF after first sorting stage");
               desc.addOptional<edm::InputTag>("ImdInputLabelEMTFNeg")->setComment("uGMT intermediate muon from neg. EMTF side after first sorting stage");
               desc.addOptional<edm::InputTag>("ImdInputLabelEMTFPos")->setComment("uGMT intermediate muon from pos. EMTF side after first sorting stage");
               desc.addOptional<edm::InputTag>("ImdInputLabelOMTFNeg")->setComment("uGMT intermediate muon from neg. OMTF side after first sorting stage");
               desc.addOptional<edm::InputTag>("ImdInputLabelOMTFPos")->setComment("uGMT intermediate muon from pos. OMTF side after first sorting stage");
            };

            virtual PackerMap getPackers(int fed, unsigned int fw) override {
               PackerMap res;

               if (fed == 1402) {
                  // Use amc_no and board id 1 for packing
                  res[{1, 1}] = {
                     PackerFactory::get()->make("stage2::RegionalMuonGMTPacker"),
                     PackerFactory::get()->make("stage2::GMTMuonPacker"),
                     PackerFactory::get()->make("stage2::IntermediateMuonPacker"),
                  };
               }

               return res;
            };

            virtual void registerProducts(edm::stream::EDProducerBase& prod) override {
               prod.produces<RegionalMuonCandBxCollection>("BMTF");
               prod.produces<RegionalMuonCandBxCollection>("OMTF");
               prod.produces<RegionalMuonCandBxCollection>("EMTF");
               prod.produces<MuonBxCollection>("Muon");
               prod.produces<MuonBxCollection>("imdMuonsBMTF");
               prod.produces<MuonBxCollection>("imdMuonsEMTFNeg");
               prod.produces<MuonBxCollection>("imdMuonsEMTFPos");
               prod.produces<MuonBxCollection>("imdMuonsOMTFNeg");
               prod.produces<MuonBxCollection>("imdMuonsOMTFPos");
            };

            virtual std::unique_ptr<UnpackerCollections> getCollections(edm::Event& e) override {
               return std::unique_ptr<UnpackerCollections>(new GMTCollections(e));
            };

            virtual UnpackerMap getUnpackers(int fed, int board, int amc, unsigned int fw) override {
               UnpackerMap res;

               auto gmt_in_unp = UnpackerFactory::get()->make("stage2::RegionalMuonGMTUnpacker");
               auto gmt_out_unp = UnpackerFactory::get()->make("stage2::MuonUnpacker");
               auto gmt_imd_unp = UnpackerFactory::get()->make("stage2::IntermediateMuonUnpacker");

               // input muons
               for (int iLink = 72; iLink < 144; iLink += 2)
                   res[iLink] = gmt_in_unp;
               // output muons
               for (int oLink = 1; oLink < 9; oLink += 2)
                   res[oLink] = gmt_out_unp;
               // internal muons
               for (int oLink = 49; oLink < 63; oLink += 2)
                   res[oLink] = gmt_imd_unp;

               return res;
            };
      };
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GMTSetup);

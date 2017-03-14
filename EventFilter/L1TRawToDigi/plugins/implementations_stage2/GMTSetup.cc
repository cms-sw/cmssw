#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/MuonUnpacker.h"

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
         desc.addOptional<edm::InputTag>("ImdInputLabelBMTF")->setComment("uGMT intermediate muon from BMTF after first sorting stage");
         desc.addOptional<edm::InputTag>("ImdInputLabelEMTFNeg")->setComment("uGMT intermediate muon from neg. EMTF side after first sorting stage");
         desc.addOptional<edm::InputTag>("ImdInputLabelEMTFPos")->setComment("uGMT intermediate muon from pos. EMTF side after first sorting stage");
         desc.addOptional<edm::InputTag>("ImdInputLabelOMTFNeg")->setComment("uGMT intermediate muon from neg. OMTF side after first sorting stage");
         desc.addOptional<edm::InputTag>("ImdInputLabelOMTFPos")->setComment("uGMT intermediate muon from pos. OMTF side after first sorting stage");
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
               PackerFactory::get()->make("stage2::IntermediateMuonPacker"),
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
         prod.produces<MuonBxCollection>("MuonSet2");
         prod.produces<MuonBxCollection>("MuonSet3");
         prod.produces<MuonBxCollection>("MuonSet4");
         prod.produces<MuonBxCollection>("MuonSet5");
         prod.produces<MuonBxCollection>("MuonSet6");
         prod.produces<MuonBxCollection>("imdMuonsBMTF");
         prod.produces<MuonBxCollection>("imdMuonsEMTFNeg");
         prod.produces<MuonBxCollection>("imdMuonsEMTFPos");
         prod.produces<MuonBxCollection>("imdMuonsOMTFNeg");
         prod.produces<MuonBxCollection>("imdMuonsOMTFPos");
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
         auto gmt_out_unp1 = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto gmt_out_unp2 = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto gmt_out_unp3 = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto gmt_out_unp4 = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto gmt_out_unp5 = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto gmt_out_unp6 = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto gmt_imd_unp = UnpackerFactory::get()->make("stage2::IntermediateMuonUnpacker");

         gmt_out_unp1->setAlgoVersion(fw);
         gmt_out_unp1->setFedNumber(fed);
         gmt_out_unp2->setAlgoVersion(fw);
         gmt_out_unp2->setFedNumber(fed);
         gmt_out_unp2->setMuonSet(2);
         gmt_out_unp3->setAlgoVersion(fw);
         gmt_out_unp3->setFedNumber(fed);
         gmt_out_unp3->setMuonSet(3);
         gmt_out_unp4->setAlgoVersion(fw);
         gmt_out_unp4->setFedNumber(fed);
         gmt_out_unp4->setMuonSet(4);
         gmt_out_unp5->setAlgoVersion(fw);
         gmt_out_unp5->setFedNumber(fed);
         gmt_out_unp5->setMuonSet(5);
         gmt_out_unp6->setAlgoVersion(fw);
         gmt_out_unp6->setFedNumber(fed);
         gmt_out_unp6->setMuonSet(6);

         // input muons
         for (int iLink = 72; iLink < 144; iLink += 2)
            res[iLink] = gmt_in_unp;

         // output muons
         // 1st set
         for (int oLink = 1; oLink < 9; oLink += 2)
            res[oLink] = gmt_out_unp1;
         // 2nd set
         for (int oLink = 9; oLink < 17; oLink += 2)
            res[oLink] = gmt_out_unp2;
         // 3rd set
         for (int oLink = 17; oLink < 25; oLink += 2)
            res[oLink] = gmt_out_unp3;
         // 4th set
         for (int oLink = 25; oLink < 33; oLink += 2)
            res[oLink] = gmt_out_unp4;
         // 5th set
         for (int oLink = 33; oLink < 41; oLink += 2)
            res[oLink] = gmt_out_unp5;
         // 6th set
         for (int oLink = 41; oLink < 49; oLink += 2)
            res[oLink] = gmt_out_unp6;

         // internal muons
         for (int oLink = 49; oLink < 63; oLink += 2)
            res[oLink] = gmt_imd_unp;

         return res;
      }
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GMTSetup);

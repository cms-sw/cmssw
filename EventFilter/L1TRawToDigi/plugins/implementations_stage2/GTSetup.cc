#include "FWCore/Framework/interface/stream/EDProducerBase.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/MuonUnpacker.h"

#include "GTSetup.h"

namespace l1t {
   namespace stage2 {
      std::unique_ptr<PackerTokens>
      GTSetup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
      {
         return std::unique_ptr<PackerTokens>(new GTTokens(cfg, cc));
      }

      void
      GTSetup::fillDescription(edm::ParameterSetDescription& desc)
      {
         desc.addOptional<edm::InputTag>("GtInputTag")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("ExtInputTag")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("MuonInputTag")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("EGammaInputTag")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("JetInputTag")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("TauInputTag")->setComment("for stage2");
         desc.addOptional<edm::InputTag>("EtSumInputTag")->setComment("for stage2");
      }

      PackerMap
      GTSetup::getPackers(int fed, unsigned int fw)
      {
         PackerMap res;

         if (fed == 1404) { 
            // Use board id 1 for packing
            res[{1, 1}] = {

               PackerFactory::get()->make("stage2::GTMuonPacker"),
               PackerFactory::get()->make("stage2::GTEGammaPacker"),
               PackerFactory::get()->make("stage2::GTEtSumPacker"),
               PackerFactory::get()->make("stage2::GTJetPacker"),
               PackerFactory::get()->make("stage2::GTTauPacker"),
               PackerFactory::get()->make("stage2::GlobalAlgBlkPacker"),
               PackerFactory::get()->make("stage2::GlobalExtBlkPacker")
            };

         }

         return res;
      }

      void
      GTSetup::registerProducts(edm::stream::EDProducerBase& prod)
      {

         prod.produces<MuonBxCollection>("Muon");
         prod.produces<EGammaBxCollection>("EGamma");
         prod.produces<EtSumBxCollection>("EtSum");
         prod.produces<JetBxCollection>("Jet");
         prod.produces<TauBxCollection>("Tau");
         prod.produces<GlobalAlgBlkBxCollection>();
         prod.produces<GlobalExtBlkBxCollection>();

      }

      std::unique_ptr<UnpackerCollections>
      GTSetup::getCollections(edm::Event& e)
      {
         return std::unique_ptr<UnpackerCollections>(new GTCollections(e));
      }

      UnpackerMap
      GTSetup::getUnpackers(int fed, int board, int amc, unsigned int fw)
      {

         auto muon_unp = static_pointer_cast<l1t::stage2::MuonUnpacker>(UnpackerFactory::get()->make("stage2::MuonUnpacker"));
         auto egamma_unp = UnpackerFactory::get()->make("stage2::EGammaUnpacker");
         auto etsum_unp = UnpackerFactory::get()->make("stage2::EtSumUnpacker");
         auto jet_unp = UnpackerFactory::get()->make("stage2::JetUnpacker");
         auto tau_unp = UnpackerFactory::get()->make("stage2::TauUnpacker");
         auto alg_unp = UnpackerFactory::get()->make("stage2::GlobalAlgBlkUnpacker");
         auto ext_unp = UnpackerFactory::get()->make("stage2::GlobalExtBlkUnpacker");

         muon_unp->setAlgoVersion(fw);
         muon_unp->setFedNumber(fed);

         UnpackerMap res;

         if (fed == 1404) {

            //only unpack first uGT board for the inputs (single copy)
            if(amc == 1) { 
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
               res[30] = ext_unp;
            }

            //From tx buffers
            res[33]  = alg_unp;
            res[35]  = alg_unp;
            res[37]  = alg_unp;
            res[39]  = alg_unp;
            res[41]  = alg_unp;
            res[43] = alg_unp;
            res[45] = alg_unp;
            res[47] = alg_unp;
            res[49] = alg_unp;		 

         }

         return res;
      }
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::GTSetup);

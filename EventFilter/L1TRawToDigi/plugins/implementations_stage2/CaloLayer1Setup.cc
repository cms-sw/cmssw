#include "FWCore/Framework/interface/stream/EDProducerBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "EventFilter/L1TRawToDigi/plugins/PackerFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/PackingSetupFactory.h"
#include "EventFilter/L1TRawToDigi/plugins/UnpackerFactory.h"

#include "CaloLayer1Setup.h"

namespace l1t {
   namespace stage2 {
      std::unique_ptr<PackerTokens>
      CaloLayer1Setup::registerConsumes(const edm::ParameterSet& cfg, edm::ConsumesCollector& cc)
      {
         return std::unique_ptr<PackerTokens>(new CaloLayer1Tokens(cfg, cc));
      }

      void
      CaloLayer1Setup::fillDescription(edm::ParameterSetDescription& desc)
      {
         desc.add<edm::InputTag>("ecalDigis");
         desc.add<edm::InputTag>("hcalDigis");
         desc.add<edm::InputTag>("caloRegions");
      }

      PackerMap
      CaloLayer1Setup::getPackers(int fed, unsigned int fw)
      {
         PackerMap res;

         if (fed == 1354) {
            // AMC #, board #
            res[{ 2,  3}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 3,  4}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 5,  5}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 8,  6}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 9,  7}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{11,  8}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
         }
         else if (fed == 1356) {
            // AMC #, board #
            res[{ 2, 15}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 3, 16}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 5, 17}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 8,  0}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 9,  1}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{11,  2}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
         }
         else if (fed == 1358) {
            // AMC #, board #
            res[{ 2,  9}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 3, 10}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 5, 11}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 8, 12}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{ 9, 13}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
            res[{11, 14}] = { PackerFactory::get()->make("stage2::CaloLayer1Packer") };
         }

         return res;
      }

      void
      CaloLayer1Setup::registerProducts(edm::stream::EDProducerBase& prod)
      {
         prod.produces<EcalTrigPrimDigiCollection>();
         prod.produces<HcalTrigPrimDigiCollection>();
         prod.produces<L1CaloRegionCollection>();
      }

      std::unique_ptr<UnpackerCollections>
      CaloLayer1Setup::getCollections(edm::Event& e)
      {
         return std::unique_ptr<UnpackerCollections>(new CaloLayer1Collections(e));
      }

      UnpackerMap
      CaloLayer1Setup::getUnpackers(int fed, int board, int amc, unsigned int fw)
      {
         auto ctp7_unp = UnpackerFactory::get()->make("stage2::CaloLayer1Unpacker");
         // if (fw >= 0xa110ca7e) {
         //    ctp7_unp = UnpackerFactory::get()->make("stage2::CaloLayer1Unpacker_v2");
         // }

         UnpackerMap res;
         LogDebug("L1T") << "CaloLayer1Setup: about to pick an unpacker for fed " << fed << " board " << board << " amc " << amc << " fw 0x" << std::hex << fw;
         if (fed == 1354 || fed == 1356 || fed == 1358) {
            if ( board < 18 && board != 4 ) {
               res[0] = ctp7_unp;
            }
         }

         return res;
      }
   }
}

DEFINE_L1T_PACKING_SETUP(l1t::stage2::CaloLayer1Setup);

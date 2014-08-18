// -*- C++ -*-
//
// Package:    EventFilter/L1TRawToDigi
// Class:      L1TDigiToRaw
// 
/**\class L1TDigiToRaw L1TDigiToRaw.cc EventFilter/L1TRawToDigi/plugins/L1TDigiToRaw.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Matthias Wolf
//         Created:  Mon, 10 Feb 2014 14:29:40 GMT
//
//

// system include files
#include <iomanip>
#include <memory>

// user include files
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "EventFilter/L1TRawToDigi/interface/PackerFactory.h"

namespace l1t {
   class L1TDigiToRaw : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
      public:
         explicit L1TDigiToRaw(const edm::ParameterSet&);
         ~L1TDigiToRaw();

         static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

         using edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks>::consumes;

      private:
         virtual void beginJob() override;
         virtual void produce(edm::Event&, const edm::EventSetup&) override;
         virtual void endJob() override;

         virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {};
         virtual void endRun(edm::Run const&, edm::EventSetup const&) override {};
         virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {};
         virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {};

         // ----------member data ---------------------------
         // FIXME is actually fixed by the firmware version
         static const unsigned MAX_BLOCKS = 256;

         edm::InputTag inputLabel_;
         int evtType_;
         int fedId_;
         unsigned fwId_;

         PackerList packers_;

         // header and trailer sizes in chars
         int slinkHeaderSize_;
         int slinkTrailerSize_;
         int amcHeaderSize_;
         int amcTrailerSize_;
   };
}

namespace l1t {
   L1TDigiToRaw::L1TDigiToRaw(const edm::ParameterSet& config) :
      inputLabel_(config.getParameter<edm::InputTag>("InputLabel")),
      fedId_(config.getParameter<int>("FedId"))
   {
      // Register products
      produces<FEDRawDataCollection>();

      fwId_ = config.getParameter<unsigned int>("FWId");
      evtType_ = config.getUntrackedParameter<int>("eventType", 1);

      auto cc = edm::ConsumesCollector(consumesCollector());

      auto packer_cfg = config.getParameterSet("packers");
      auto packer_names = packer_cfg.getParameterNames();

      for (const auto& name: packer_names) {
         const auto& pset = packer_cfg.getParameterSet(name);
         auto factory = std::auto_ptr<BasePackerFactory>(PackerFactory::get()->makePackerFactory(pset, cc));
         auto packer_list = factory->create(fwId_, fedId_);
         packers_.insert(packers_.end(), packer_list.begin(), packer_list.end());
      }

      slinkHeaderSize_ = config.getUntrackedParameter<int>("lenSlinkHeader", 16);
      slinkTrailerSize_ = config.getUntrackedParameter<int>("lenSlinkTrailer", 16);
      amcHeaderSize_ = config.getUntrackedParameter<int>("lenAMCHeader", 12);
      amcTrailerSize_ = config.getUntrackedParameter<int>("lenAMCTrailer", 8);
   }


   L1TDigiToRaw::~L1TDigiToRaw()
   {
   }

   inline unsigned char *
   push(unsigned char * ptr, uint32_t word)
   {
      ptr[0] = word & 0xFF;
      ptr[1] = (word >> 8) & 0xFF;
      ptr[2] = (word >> 16) & 0xFF;
      ptr[3] = (word >> 24) & 0xFF;
      return ptr + 4;
   }

   // ------------ method called to produce the data  ------------
   void
   L1TDigiToRaw::produce(edm::Event& event, const edm::EventSetup& setup)
   {
      using namespace edm;

      Blocks blocks;

      for (auto& packer: packers_) {
         auto pblocks = packer->pack(event);
         blocks.insert(blocks.end(), pblocks.begin(), pblocks.end());
      }

      std::auto_ptr<FEDRawDataCollection> raw_coll(new FEDRawDataCollection());
      FEDRawData& fed_data = raw_coll->FEDData(fedId_);

      unsigned int size = slinkHeaderSize_ + slinkTrailerSize_ + amcHeaderSize_ + amcTrailerSize_;
      unsigned int words = 0;
      for (const auto& block: blocks)
         // add one for the block header
         words += block.load.size() + 1;
      size += words * 4;
      // add padding to get a full number of 64-bit words
      int padding = (size % 8 == 0) ? 8 : 8 - size % 8;
      size += padding;
      fed_data.resize(size);
      unsigned char * payload = fed_data.data();
      auto payload_start = payload;

      auto bxId = event.bunchCrossing();
      auto evtId = event.id().event();

      FEDHeader header(payload);
      header.set(payload, evtType_, evtId, bxId, fedId_);

      payload += slinkHeaderSize_;

      // create the header
      payload = push(payload, 0);
      payload = push(payload, 0);
      payload = push(payload, (fwId_ << 24) | (words << 8)); // FW ID, payload size (words)

      for (const auto& block: blocks) {
         payload = push(payload, (block.id << 24) | (block.load.size() << 16));
         for (const auto& word: block.load)
            payload = push(payload, word);
      }

      payload += amcTrailerSize_;
      payload += padding;

      FEDTrailer trailer(payload);
      trailer.set(payload, size / 8, evf::compute_crc(payload_start, size), 0, 0);

      event.put(raw_coll);
   }

   // ------------ method called once each job just before starashtting event loop  ------------
   void 
   L1TDigiToRaw::beginJob()
   {
   }

   // ------------ method called once each job just after ending the event loop  ------------
   void 
   L1TDigiToRaw::endJob() {
   }

   // ------------ method called when starting to processes a run  ------------
   /*
   void
   L1TDigiToRaw::beginRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method called when ending the processing of a run  ------------
   /*
   void
   L1TDigiToRaw::endRun(edm::Run const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method called when starting to processes a luminosity block  ------------
   /*
   void
   L1TDigiToRaw::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method called when ending the processing of a luminosity block  ------------
   /*
   void
   L1TDigiToRaw::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
   {
   }
   */
    
   // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
   void
   L1TDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
     //The following says we do not know what parameters are allowed so do no validation
     // Please change this to state exactly what you do use, even if it is no parameters
     edm::ParameterSetDescription desc;
     desc.setUnknown();
     descriptions.addDefault(desc);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TDigiToRaw);

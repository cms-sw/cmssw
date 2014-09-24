// -*- C++ -*-
//
// Package:    EventFilter/L1TRawToDigi
// Class:      L1TRawToDigi
//
/**\class L1TRawToDigi L1TRawToDigi.cc EventFilter/L1TRawToDigi/plugins/L1TRawToDigi.cc

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
#include <memory>

#define EDM_ML_DEBUG 1

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerFactory.h"

namespace l1t {
   class BaseUnpackerFactory;

   class L1TRawToDigi : public edm::one::EDProducer<edm::one::SharedResources, edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
      public:
         explicit L1TRawToDigi(const edm::ParameterSet&);
         ~L1TRawToDigi();

         static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

      private:
         virtual void produce(edm::Event&, const edm::EventSetup&) override;

         virtual void beginRun(edm::Run const&, edm::EventSetup const&) override {};
         virtual void endRun(edm::Run const&, edm::EventSetup const&) override {};
         virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {};
         virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {};

         // ----------member data ---------------------------
         edm::EDGetTokenT<FEDRawDataCollection> fedData_;
         int fedId_;
         std::vector<std::auto_ptr<BaseUnpackerFactory>> factories_;

         std::string product_;

         // header and trailer sizes in chars
         int slinkHeaderSize_;
         int slinkTrailerSize_;
         int amcHeaderSize_;
         int amcTrailerSize_;
   };
}

namespace l1t {
   L1TRawToDigi::L1TRawToDigi(const edm::ParameterSet& config) :
      fedId_(config.getParameter<int>("FedId")),
      product_("l1t::L1T")
   {
      fedData_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("InputLabel"));

      UnpackerCollectionsProducesFactory::get()->makeUnpackerCollectionsProduces(product_ + "CollectionsProduces", *this);

      auto factory_names = config.getParameter<std::vector<std::string>>("Unpackers");
      for (const auto& name: factory_names)
         factories_.push_back(UnpackerFactory::get()->makeUnpackerFactory(name));

      slinkHeaderSize_ = config.getUntrackedParameter<int>("lenSlinkHeader", 16);
      slinkTrailerSize_ = config.getUntrackedParameter<int>("lenSlinkTrailer", 16);
      amcHeaderSize_ = config.getUntrackedParameter<int>("lenAMCHeader", 12);
      amcTrailerSize_ = config.getUntrackedParameter<int>("lenAMCTrailer", 8);
   }


   L1TRawToDigi::~L1TRawToDigi()
   {
   }


   //
   // member functions
   //

   // ------------ method called to produce the data  ------------
   void
   L1TRawToDigi::produce(edm::Event& event, const edm::EventSetup& setup)
   {
      using namespace edm;

      std::auto_ptr<UnpackerCollections> coll(UnpackerCollectionsFactory::get()->makeUnpackerCollections(product_ + "Collections", event));

      edm::Handle<FEDRawDataCollection> feds;
      event.getByToken(fedData_, feds);

      if (!feds.isValid()) {
         LogError("L1T") << "Cannot unpack: no collection found";
         return;
      }

      LogInfo("L1T") << "Found FEDRawDataCollection";

      const FEDRawData& l1tRcd = feds->FEDData(fedId_);

      if ((int) l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amcHeaderSize_ + amcTrailerSize_) {
         LogError("L1T") << "Cannot unpack: empty/invalid L1T raw data (size = "
            << l1tRcd.size() << "). Returning empty collections!";
         return;
      }

      const unsigned char *data = l1tRcd.data();
      unsigned idx = slinkHeaderSize_;

      FEDHeader header(data);

      if (header.check()) {
         LogDebug("L1T") << "Found SLink header:"
            << " Trigger type " << header.triggerType()
            << " L1 event ID " << header.lvl1ID()
            << " BX Number " << header.bxID()
            << " FED source " << header.sourceID()
            << " FED version " << header.version();
      } else {
         LogWarning("L1T") << "Did not find a SLink header!";
      }

      FEDTrailer trailer(data + (l1tRcd.size() - slinkTrailerSize_));

      if (trailer.check()) {
         LogDebug("L1T") << "Found SLink trailer:"
            << " Length " << trailer.lenght()
            << " CRC " << trailer.crc()
            << " Status " << trailer.evtStatus()
            << " Throttling bits " << trailer.ttsBits();
      } else {
         LogWarning("L1T") << "Did not find a SLink trailer!";
      }

      // Extract header data
      uint32_t event_id = pop(data, idx) & 0xFFFFFF;

      //      uint32_t id = pop(data, idx);
      uint32_t id = pop(data, idx);
      uint32_t bx_id = (id >> 16) & 0xFFF;
      uint32_t orbit_id = (id >> 4) & 0x1F;
      uint32_t board_id = id & 0xF;

      id = pop(data, idx);
      uint32_t fw_id = (id >> 24) & 0xFF;
      uint32_t payload_size = (id >> 8) & 0xFFFF;
      uint32_t event_type = id & 0xFF;

      LogDebug("L1T") << "Found AMC13 header: Event Number " << event_id
                      << " Board ID " << board_id
                      << " Orbit Number " << orbit_id
                      << " BX Number " << bx_id
                      << " FW version " << fw_id
                      << " Event Type " << event_type
                      << " Payload size " << payload_size;

      if (l1tRcd.size() < payload_size * 4 + amcHeaderSize_ + amcTrailerSize_) {
         LogError("L1T") << "Cannot unpack: invalid L1T raw data size in header (size = "
            << l1tRcd.size() << ", expected "
            << payload_size * 4 + amcHeaderSize_ + amcTrailerSize_
            << " + padding). Returning empty collections!";
         return;
      }

      unsigned fw = fw_id;

      UnpackerMap unpackers;
      for (auto& f: factories_) {
        for (const auto& up: f->create(fw, fedId_)) {
            unpackers.insert(up);
         }
      }

      auto payload_end = idx + payload_size * 4;
      for (unsigned int b = 0; idx < payload_end; ++b) {
         // Parse block
         uint32_t block_hdr = pop(data, idx);
         uint32_t block_id = (block_hdr >> 24) & 0xFF;
         uint32_t block_size = (block_hdr >> 16) & 0xFF;

         LogDebug("L1T") << "Found MP7 header: block ID " << block_id << " block size " << block_size;

         auto unpacker = unpackers.find(block_id);
         if (unpacker == unpackers.end()) {
            LogWarning("L1T") << "Cannot find an unpacker for block ID "
               << block_id << ", FED ID " << fedId_ << ", and FW ID "
               << fw << "!";
            // TODO Handle error
         } else if (!unpacker->second->unpack(block_id, block_size, data + idx, coll.get())) {
            LogWarning("L1T") << "Error unpacking data for block ID "
               << block_id << ", FED ID " << fedId_ << ", and FW ID "
               << fw << "!";
            // TODO Handle error
         }

         // Advance index by block size (header does this in pop())
         idx += block_size * 4;
      }
   }

   // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
   void
   L1TRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
     //The following says we do not know what parameters are allowed so do no validation
     // Please change this to state exactly what you do use, even if it is no parameters
     edm::ParameterSetDescription desc;
     desc.setUnknown();
     descriptions.addDefault(desc);
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(l1t::L1TRawToDigi);

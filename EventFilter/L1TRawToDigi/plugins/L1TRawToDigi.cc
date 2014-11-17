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

#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"
#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "EventFilter/L1TRawToDigi/interface/PackingSetup.h"

namespace l1t {
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
         int fwId_;

         std::auto_ptr<PackingSetup> prov_;

         // header and trailer sizes in chars
         int slinkHeaderSize_;
         int slinkTrailerSize_;
         int amcHeaderSize_;
         int amcTrailerSize_;
         int amc13HeaderSize_;
         int amc13TrailerSize_;
   };
}

std::ostream & operator<<(std::ostream& o, const l1t::BlockHeader& h) {
   o << "L1T Block Header " << h.getID() << " with size " << h.getSize();
   return o;
};

namespace l1t {
   L1TRawToDigi::L1TRawToDigi(const edm::ParameterSet& config) :
      fedId_(config.getParameter<int>("FedId")),
      fwId_(config.getUntrackedParameter<int>("FWId", -1))
   {
      fedData_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("InputLabel"));

      prov_ = PackingSetupFactory::get()->make(config.getParameter<std::string>("Setup"));
      prov_->registerProducts(*this);

      slinkHeaderSize_ = config.getUntrackedParameter<int>("lenSlinkHeader", 16);
      slinkTrailerSize_ = config.getUntrackedParameter<int>("lenSlinkTrailer", 16);
      amcHeaderSize_ = config.getUntrackedParameter<int>("lenAMCHeader", 8);
      amcTrailerSize_ = config.getUntrackedParameter<int>("lenAMCTrailer", 0);
      amc13HeaderSize_ = config.getUntrackedParameter<int>("lenAMC13Header", 8);
      amc13TrailerSize_ = config.getUntrackedParameter<int>("lenAMC13Trailer", 8);
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

      std::unique_ptr<UnpackerCollections> coll = prov_->getCollections(event);

      edm::Handle<FEDRawDataCollection> feds;
      event.getByToken(fedData_, feds);

      if (!feds.isValid()) {
         LogError("L1T") << "Cannot unpack: no collection found";
         return;
      }

      const FEDRawData& l1tRcd = feds->FEDData(fedId_);

      LogDebug("L1T") << "Found FEDRawDataCollection with ID " << fedId_ << " and size " << l1tRcd.size();

      if ((int) l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ + amcHeaderSize_ + amcTrailerSize_) {
	//LogError("L1T") << "Cannot unpack: empty/invalid L1T raw data (size = "
	//   << l1tRcd.size() << ") for ID " << fedId_ << ". Returning empty collections!";
         return;
      }

      const unsigned char *data = l1tRcd.data();
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

      amc13::Packet packet;
      if (!packet.parse(
               (const uint64_t*) (data + slinkHeaderSize_),
               (l1tRcd.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8)) {
         LogError("L1T")
            << "Could not extract AMC13 Packet.";
         return;
      }

      for (auto& amc: packet.payload()) {
         auto payload64 = amc.data();
         const uint32_t * payload = (const uint32_t*) payload64.get();
         const uint32_t * end = payload + (amc.size() * 2);

         // TODO this skips the still to be added MP7 header containing the
         // firmware version
         unsigned fw = 0;
         payload++;

         // Let parameterset value override FW version
         if (fwId_ > 0)
            fw = fwId_;

         unsigned board = amc.header().getBoardID();

         auto unpackers = prov_->getUnpackers(fedId_, board, fw);

         while (payload != end) {
            BlockHeader block_hdr(payload++);

            /* LogDebug("L1T") << "Found " << block_hdr; */
            //LogDebug("L1T") << "Found block " << block_hdr.getID() << " with size " << block_hdr.getSize();

            if (end - payload < block_hdr.getSize()) {
               LogError("L1T")
                  << "Expecting a block size of " << block_hdr.getSize()
                  << " but only " << (end - payload) << " words remaining";
               return;
            }

            Block block(block_hdr, payload, payload + block_hdr.getSize());

            auto unpacker = unpackers.find(block_hdr.getID());
            if (unpacker == unpackers.end()) {
	      //LogWarning("L1T") << "Cannot find an unpacker for block ID "
	      //  << block_hdr.getID() << ", FED ID " << fedId_ << ", and FW ID "
	      //  << fw << "!";
               // TODO Handle error
            } else if (!unpacker->second->unpack(block, coll.get())) {
               LogWarning("L1T") << "Error unpacking data for block ID "
                  << block_hdr.getID() << ", FED ID " << fedId_ << ", and FW ID "
                  << fw << "!";
               // TODO Handle error
            }

            payload += block_hdr.getSize();
         }
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

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(L1TRawToDigi);

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
#include <iostream>
#include <iomanip>
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

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
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
         std::vector<int> fedIds_;
         int fwId_;

         std::auto_ptr<PackingSetup> prov_;

         // header and trailer sizes in chars
         int slinkHeaderSize_;
         int slinkTrailerSize_;
         int amcHeaderSize_;
         int amcTrailerSize_;
         int amc13HeaderSize_;
         int amc13TrailerSize_;

         bool ctp7_mode_;
         bool debug_;
   };
}

std::ostream & operator<<(std::ostream& o, const l1t::BlockHeader& h) {
   o << "L1T Block Header " << h.getID() << " with size " << h.getSize();
   return o;
};

namespace l1t {
   L1TRawToDigi::L1TRawToDigi(const edm::ParameterSet& config) :
      fwId_(config.getUntrackedParameter<int>("FWId", -1)),
      ctp7_mode_(config.getUntrackedParameter<bool>("CTP7", false))
   {
      fedData_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("InputLabel"));

      if (config.exists("FedId") and config.exists("FedIds")) {
         throw edm::Exception(edm::errors::Configuration, "PSet")
            << "Cannot have FedId and FedIds as parameter at the same time";
      } else if (config.exists("FedId")) {
         fedIds_ = {config.getParameter<int>("FedId")};
      } else {
         fedIds_ = config.getParameter<std::vector<int>>("FedIds");
      }

      prov_ = PackingSetupFactory::get()->make(config.getParameter<std::string>("Setup"));
      prov_->registerProducts(*this);

      slinkHeaderSize_ = config.getUntrackedParameter<int>("lenSlinkHeader", 8);
      slinkTrailerSize_ = config.getUntrackedParameter<int>("lenSlinkTrailer", 8);
      amcHeaderSize_ = config.getUntrackedParameter<int>("lenAMCHeader", 8);
      amcTrailerSize_ = config.getUntrackedParameter<int>("lenAMCTrailer", 0);
      amc13HeaderSize_ = config.getUntrackedParameter<int>("lenAMC13Header", 8);
      amc13TrailerSize_ = config.getUntrackedParameter<int>("lenAMC13Trailer", 8);

      debug_ = config.getUntrackedParameter<bool>("debug", false);
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

      for (const auto& fedId: fedIds_) {
         const FEDRawData& l1tRcd = feds->FEDData(fedId);

         LogDebug("L1T") << "Found FEDRawDataCollection with ID " << fedId << " and size " << l1tRcd.size();

         if ((int) l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ + amcHeaderSize_ + amcTrailerSize_) {
            LogError("L1T") << "Cannot unpack: empty/invalid L1T raw data (size = "
               << l1tRcd.size() << ") for ID " << fedId << ". Returning empty collections!";
	    continue;
            //return;
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
            const uint32_t * start = (const uint32_t*) payload64.get();
            const uint32_t * end = start + (amc.size() * 2);

            std::auto_ptr<Payload> payload;
            if (ctp7_mode_) {
               LogDebug("L1T") << "Using CTP7 mode";
               payload.reset(new CTP7Payload(start, end));
            } else {
               LogDebug("L1T") << "Using MP7 mode";
               payload.reset(new MP7Payload(start, end));
            }
            unsigned fw = payload->getFirmwareId();

            // Let parameterset value override FW version
            if (fwId_ > 0)
               fw = fwId_;

            unsigned board = amc.blockHeader().getBoardID();
            unsigned amc_no = amc.blockHeader().getAMCNumber();

            auto unpackers = prov_->getUnpackers(fedId, board, amc_no, fw);

            // getBlock() returns a non-null auto_ptr on success
            std::auto_ptr<Block> block;
            while ((block = payload->getBlock()).get()) {
               // skip empty filler blocks
               if ((block->header().getID() == 0 and block->header().getSize() == 0) or block->header().raw() == 0xffffffff)
                  continue;

               if (debug_) {
                  std::cout << ">>> block to unpack <<<" << std::endl
                     << "hdr:  " << std::hex << std::setw(8) << block->header().raw() << std::endl;
                  for (const auto& word: block->payload()) {
                     std::cout << "data: " << std::hex << std::setw(8) << word << std::endl;
                  }
               }

               auto unpacker = unpackers.find(block->header().getID());

               block->amc(amc.header());

               if (unpacker == unpackers.end()) {
                  LogDebug("L1T") << "Cannot find an unpacker for block ID "
                     << block->header().getID() << ", AMC # " << amc_no
                     << ", board ID " << board << ", FED ID " << fedId
                     << ", and FW ID " << fw << "!";
                  // TODO Handle error
               } else if (!unpacker->second->unpack(*block, coll.get())) {
                  LogDebug("L1T") << "Error unpacking data for block ID "
                     << block->header().getID() << ", AMC # " << amc_no
                     << ", board ID " << board << ", FED ID " << fedId
                     << ", and FW ID " << fw << "!";
                  // TODO Handle error
               }
            }
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

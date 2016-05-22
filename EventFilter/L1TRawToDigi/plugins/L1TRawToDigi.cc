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
#include "FWCore/Framework/interface/stream/EDProducer.h"
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
   class L1TRawToDigi : public edm::stream::EDProducer<> {
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
         unsigned int fwId_;
         bool fwOverride_;

         std::auto_ptr<PackingSetup> prov_;

         // header and trailer sizes in chars
         int slinkHeaderSize_;
         int slinkTrailerSize_;
         int amcHeaderSize_;
         int amcTrailerSize_;
         int amc13HeaderSize_;
         int amc13TrailerSize_;

         bool ctp7_mode_;
         bool mtf7_mode_;
         bool debug_;
         int warns_;
   };
}

std::ostream & operator<<(std::ostream& o, const l1t::BlockHeader& h) {
   o << "L1T Block Header " << h.getID() << " with size " << h.getSize();
   return o;
};

namespace l1t {
   L1TRawToDigi::L1TRawToDigi(const edm::ParameterSet& config) :
      fedIds_(config.getParameter<std::vector<int>>("FedIds")),
      fwId_(config.getParameter<unsigned int>("FWId")),
      fwOverride_(config.getParameter<bool>("FWOverride")),
      ctp7_mode_(config.getUntrackedParameter<bool>("CTP7")),
      mtf7_mode_(config.getUntrackedParameter<bool>("MTF7"))
   {
      fedData_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("InputLabel"));

      if (ctp7_mode_ and mtf7_mode_) {
	throw cms::Exception("L1TRawToDigi") << "Can only use one unpacking mode concurrently!";
      }

      prov_ = PackingSetupFactory::get()->make(config.getParameter<std::string>("Setup"));
      prov_->registerProducts(*this);

      slinkHeaderSize_ = config.getUntrackedParameter<int>("lenSlinkHeader");
      slinkTrailerSize_ = config.getUntrackedParameter<int>("lenSlinkTrailer");
      amcHeaderSize_ = config.getUntrackedParameter<int>("lenAMCHeader");
      amcTrailerSize_ = config.getUntrackedParameter<int>("lenAMCTrailer");
      amc13HeaderSize_ = config.getUntrackedParameter<int>("lenAMC13Header");
      amc13TrailerSize_ = config.getUntrackedParameter<int>("lenAMC13Trailer");

      debug_ = config.getUntrackedParameter<bool>("debug");
      warns_ = 0;
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
         LogError("L1T") << "Cannot unpack: no FEDRawDataCollection found";
         return;
      }

      for (const auto& fedId: fedIds_) {
         const FEDRawData& l1tRcd = feds->FEDData(fedId);

         LogDebug("L1T") << "Found FEDRawDataCollection with ID " << fedId << " and size " << l1tRcd.size();

         if ((int) l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ + amcHeaderSize_ + amcTrailerSize_) {
	   if (l1tRcd.size() > 0) {
            LogError("L1T") << "Cannot unpack: invalid L1T raw data (size = "
               << l1tRcd.size() << ") for ID " << fedId << ". Returning empty collections!";
	   } else if (warns_ < 5) {
	     warns_++;
	     LogWarning("L1T") << "Cannot unpack: empty L1T raw data (size = "
			       << l1tRcd.size() << ") for ID " << fedId << ". Returning empty collections!";
	   }
            continue;
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

         // FIXME Hard-coded firmware version for first 74x MC campaigns.
         // Will account for differences in the AMC payload, MP7 payload,
         // and unpacker setup.
	 bool legacy_mc = fwOverride_ && ((fwId_ >> 24) == 0xff);

         amc13::Packet packet;
         if (!packet.parse(
                  (const uint64_t*) data,
                  (const uint64_t*) (data + slinkHeaderSize_),
                  (l1tRcd.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8,
                  header.lvl1ID(),
                  header.bxID(),
                  legacy_mc,
		  mtf7_mode_)) {
            LogError("L1T")
               << "Could not extract AMC13 Packet.";
            return;
         }

         for (auto& amc: packet.payload()) {
	   if (amc.size() == 0)
	     continue;

            auto payload64 = amc.data();
            const uint32_t * start = (const uint32_t*) payload64.get();
            // Want to have payload size in 32 bit words, but AMC measures
            // it in 64 bit words → factor 2.
            const uint32_t * end = start + (amc.size() * 2);

            std::auto_ptr<Payload> payload;
            if (ctp7_mode_) {
               LogDebug("L1T") << "Using CTP7 mode";
               payload.reset(new CTP7Payload(start, end));
            } else if (mtf7_mode_) {
               LogDebug("L1T") << "Using MTF7 mode";
               payload.reset(new MTF7Payload(start, end));
	    } else {
               LogDebug("L1T") << "Using MP7 mode";
               payload.reset(new MP7Payload(start, end, legacy_mc));
            }
            unsigned fw = payload->getAlgorithmFWVersion();

            // Let parameterset value override FW version
            if (fwOverride_)
               fw = fwId_;

            unsigned board = amc.blockHeader().getBoardID();
            unsigned amc_no = amc.blockHeader().getAMCNumber();

            auto unpackers = prov_->getUnpackers(fedId, board, amc_no, fw);

            // getBlock() returns a non-null auto_ptr on success
            std::auto_ptr<Block> block;
            while ((block = payload->getBlock()).get()) {
               if (debug_) {
                  std::cout << ">>> block to unpack <<<" << std::endl
                     << "hdr:  " << std::hex << std::setw(8) << std::setfill('0') << block->header().raw() << std::dec
                     << " (ID " << block->header().getID() << ", size " << block->header().getSize()
                     << ", CapID 0x" << std::hex << std::setw(2) << std::setfill('0') << block->header().getCapID()
			    << ")" << std::dec << std::endl;
                  for (const auto& word: block->payload()) {
		    if (debug_)
		      std::cout << "data: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
		  }
               }

               auto unpacker = unpackers.find(block->header().getID());

               block->amc(amc.header());

               if (unpacker == unpackers.end()) {
                  LogDebug("L1T") << "Cannot find an unpacker for"
                     << "\n\tblock: ID " << block->header().getID() << ", size " << block->header().getSize()
                     << "\n\tAMC: # " << amc_no << ", board ID 0x" << std::hex << board << std::dec
                     << "\n\tFED ID " << fedId << ", and FW ID " << fw;
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
     edm::ParameterSetDescription desc;
     // These parameters are part of the L1T/HLT interface, avoid changing if possible:
     desc.add<std::vector<int>>("FedIds", {})->setComment("required parameter:  default value is invalid");
     desc.add<std::string>("Setup", "")->setComment("required parameter:  default value is invalid");
     // These parameters have well defined  default values and are not currently 
     // part of the L1T/HLT interface.  They can be cleaned up or updated at will:     
     desc.add<unsigned int>("FWId",0)->setComment("Ignored unless FWOverride is true.  Calo Stage1:  32 bits: if the first eight bits are 0xff, will read the 74x MC format.\n");
     desc.add<bool>("FWOverride", false)->setComment("Firmware version should be taken as FWId parameters");
     desc.addUntracked<bool>("CTP7", false);
     desc.addUntracked<bool>("MTF7", false);
     desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
     desc.addUntracked<int>("lenSlinkHeader", 8);
     desc.addUntracked<int>("lenSlinkTrailer", 8);
     desc.addUntracked<int>("lenAMCHeader", 8);
     desc.addUntracked<int>("lenAMCTrailer", 0);
     desc.addUntracked<int>("lenAMC13Header", 8);
     desc.addUntracked<int>("lenAMC13Trailer", 8);
     desc.addUntracked<bool>("debug", false)->setComment("turn on verbose output");
     descriptions.add("l1tRawToDigi", desc);
   }
}

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(L1TRawToDigi);

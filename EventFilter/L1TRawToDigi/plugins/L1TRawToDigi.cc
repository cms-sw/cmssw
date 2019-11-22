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

#include "PackingSetupFactory.h"

#include "EventFilter/L1TRawToDigi/plugins/implementations_stage2/L1TStage2Layer2Constants.h"

namespace l1t {
  class L1TRawToDigi : public edm::stream::EDProducer<> {
  public:
    explicit L1TRawToDigi(const edm::ParameterSet&);
    ~L1TRawToDigi() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    void beginRun(edm::Run const&, edm::EventSetup const&) override{};
    void endRun(edm::Run const&, edm::EventSetup const&) override{};
    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override{};
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override{};

    // ----------member data ---------------------------
    edm::EDGetTokenT<FEDRawDataCollection> fedData_;
    std::vector<int> fedIds_;
    unsigned int minFeds_;
    unsigned int fwId_;
    unsigned int dmxFwId_;
    bool fwOverride_;

    std::unique_ptr<PackingSetup> prov_;

    // header and trailer sizes in chars
    int slinkHeaderSize_;
    int slinkTrailerSize_;
    int amcHeaderSize_;
    int amcTrailerSize_;
    int amc13HeaderSize_;
    int amc13TrailerSize_;

    bool tmtCheck_;

    bool ctp7_mode_;
    bool mtf7_mode_;
    bool debug_;
    int warnsa_;
    int warnsb_;
  };
}  // namespace l1t

std::ostream& operator<<(std::ostream& o, const l1t::BlockHeader& h) {
  o << "L1T Block Header " << h.getID() << " with size " << h.getSize();
  return o;
};

namespace l1t {
  L1TRawToDigi::L1TRawToDigi(const edm::ParameterSet& config)
      : fedIds_(config.getParameter<std::vector<int>>("FedIds")),
        minFeds_(config.getParameter<unsigned int>("MinFeds")),
        fwId_(config.getParameter<unsigned int>("FWId")),
        dmxFwId_(config.getParameter<unsigned int>("DmxFWId")),
        fwOverride_(config.getParameter<bool>("FWOverride")),
        tmtCheck_(config.getParameter<bool>("TMTCheck")),
        ctp7_mode_(config.getUntrackedParameter<bool>("CTP7")),
        mtf7_mode_(config.getUntrackedParameter<bool>("MTF7")) {
    fedData_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("InputLabel"));

    if (ctp7_mode_ and mtf7_mode_) {
      throw cms::Exception("L1TRawToDigi") << "Can only use one unpacking mode concurrently!";
    }

    prov_ = PackingSetupFactory::get()->make(config.getParameter<std::string>("Setup"));
    prov_->registerProducts(producesCollector());

    slinkHeaderSize_ = config.getUntrackedParameter<int>("lenSlinkHeader");
    slinkTrailerSize_ = config.getUntrackedParameter<int>("lenSlinkTrailer");
    amcHeaderSize_ = config.getUntrackedParameter<int>("lenAMCHeader");
    amcTrailerSize_ = config.getUntrackedParameter<int>("lenAMCTrailer");
    amc13HeaderSize_ = config.getUntrackedParameter<int>("lenAMC13Header");
    amc13TrailerSize_ = config.getUntrackedParameter<int>("lenAMC13Trailer");

    debug_ = config.getUntrackedParameter<bool>("debug");
    warnsa_ = 0;
    warnsb_ = 0;
  }

  L1TRawToDigi::~L1TRawToDigi() {}

  //
  // member functions
  //

  // ------------ method called to produce the data  ------------
  void L1TRawToDigi::produce(edm::Event& event, const edm::EventSetup& setup) {
    using namespace edm;

    std::unique_ptr<UnpackerCollections> coll = prov_->getCollections(event);

    edm::Handle<FEDRawDataCollection> feds;
    event.getByToken(fedData_, feds);

    if (!feds.isValid()) {
      LogError("L1T") << "Cannot unpack: no FEDRawDataCollection found";
      return;
    }

    unsigned valid_count = 0;
    for (const auto& fedId : fedIds_) {
      const FEDRawData& l1tRcd = feds->FEDData(fedId);

      LogDebug("L1T") << "Found FEDRawDataCollection with ID " << fedId << " and size " << l1tRcd.size();

      if ((int)l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ +
                                   amcHeaderSize_ + amcTrailerSize_) {
        if (l1tRcd.size() > 0) {
          LogError("L1T") << "Cannot unpack: invalid L1T raw data (size = " << l1tRcd.size() << ") for ID " << fedId
                          << ". Returning empty collections!";
        } else if (warnsa_ < 5) {
          warnsa_++;
          LogInfo("L1T") << "During unpacking, encountered empty L1T raw data (size = " << l1tRcd.size()
                         << ") for FED ID " << fedId << ".";
        }
        continue;
      } else {
        valid_count++;
      }

      const unsigned char* data = l1tRcd.data();
      FEDHeader header(data);

      if (header.check()) {
        LogDebug("L1T") << "Found SLink header:"
                        << " Trigger type " << header.triggerType() << " L1 event ID " << header.lvl1ID()
                        << " BX Number " << header.bxID() << " FED source " << header.sourceID() << " FED version "
                        << header.version();
      } else {
        LogWarning("L1T") << "Did not find a SLink header!";
      }

      FEDTrailer trailer(data + (l1tRcd.size() - slinkTrailerSize_));

      if (trailer.check()) {
        LogDebug("L1T") << "Found SLink trailer:"
                        << " Length " << trailer.fragmentLength() << " CRC " << trailer.crc() << " Status "
                        << trailer.evtStatus() << " Throttling bits " << trailer.ttsBits();
      } else {
        LogWarning("L1T") << "Did not find a SLink trailer!";
      }

      // FIXME Hard-coded firmware version for first 74x MC campaigns.
      // Will account for differences in the AMC payload, MP7 payload,
      // and unpacker setup.
      bool legacy_mc = fwOverride_ && ((fwId_ >> 24) == 0xff);

      amc13::Packet packet;
      if (!packet.parse((const uint64_t*)data,
                        (const uint64_t*)(data + slinkHeaderSize_),
                        (l1tRcd.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8,
                        header.lvl1ID(),
                        header.bxID(),
                        legacy_mc,
                        mtf7_mode_)) {
        LogError("L1T") << "Could not extract AMC13 Packet.";
        return;
      }

      for (auto& amc : packet.payload()) {
        if (amc.size() == 0)
          continue;

        auto payload64 = amc.data();
        const uint32_t* start = (const uint32_t*)payload64.get();
        // Want to have payload size in 32 bit words, but AMC measures
        // it in 64 bit words → factor 2.
        const uint32_t* end = start + (amc.size() * 2);

        std::unique_ptr<Payload> payload;
        if (ctp7_mode_) {
          LogDebug("L1T") << "Using CTP7 mode";
          // CTP7 uses userData in AMC header
          payload.reset(new CTP7Payload(start, end, amc.header()));
        } else if (mtf7_mode_) {
          LogDebug("L1T") << "Using MTF7 mode";
          payload.reset(new MTF7Payload(start, end));
        } else {
          LogDebug("L1T") << "Using MP7 mode";
          payload.reset(new MP7Payload(start, end, legacy_mc));
        }
        unsigned fw = payload->getAlgorithmFWVersion();
        unsigned board = amc.blockHeader().getBoardID();
        unsigned amc_no = amc.blockHeader().getAMCNumber();

        // Let parameterset value override FW version
        if (fwOverride_) {
          if (fedId == 1360)
            fw = fwId_;
          else if (fedId == 1366)
            fw = dmxFwId_;
        }

        auto unpackers = prov_->getUnpackers(fedId, board, amc_no, fw);

        // getBlock() returns a non-null unique_ptr on success
        std::unique_ptr<Block> block;
        while ((block = payload->getBlock()).get()) {
          // only unpack the Calo Layer 2 MP TMT node if it has processed this BX
          unsigned tmtId = board - l1t::stage2::layer2::mp::offsetBoardId + 1;
          unsigned bxId = header.bxID();
          unsigned unpackTMT = (!tmtCheck_ || ((tmtId - 1) == ((bxId - 1 + 3) % 9)));
          unsigned isCaloL2TMT =
              (fedId == l1t::stage2::layer2::fedId && (amc_no != l1t::stage2::layer2::demux::amcSlotNum));

          if (!isCaloL2TMT || unpackTMT) {
            if (debug_) {
              std::cout << ">>> block to unpack <<<" << std::endl
                        << "hdr:  " << std::hex << std::setw(8) << std::setfill('0') << block->header().raw()
                        << std::dec << " (ID " << block->header().getID() << ", size " << block->header().getSize()
                        << ", CapID 0x" << std::hex << std::setw(2) << std::setfill('0') << block->header().getCapID()
                        << ")" << std::dec << std::endl;
              for (const auto& word : block->payload()) {
                if (debug_)
                  std::cout << "data: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec
                            << std::endl;
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
              LogDebug("L1T") << "Error unpacking data for block ID " << block->header().getID() << ", AMC # " << amc_no
                              << ", board ID " << board << ", FED ID " << fedId << ", and FW ID " << fw << "!";
              // TODO Handle error
            }
          }
        }
      }
    }
    if (valid_count < minFeds_) {
      if (warnsb_ < 5) {
        warnsb_++;
        LogWarning("L1T") << "Unpacked " << valid_count << " non-empty FED IDs but minimum is set to " << minFeds_
                          << "\n";
      }
    }
  }

  // ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
  void L1TRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    // These parameters are part of the L1T/HLT interface, avoid changing if possible:
    desc.add<std::vector<int>>("FedIds", {})->setComment("required parameter:  default value is invalid");
    desc.add<std::string>("Setup", "")->setComment("required parameter:  default value is invalid");
    // These parameters have well defined  default values and are not currently
    // part of the L1T/HLT interface.  They can be cleaned up or updated at will:
    desc.add<unsigned int>("FWId", 0)->setComment(
        "Ignored unless FWOverride is true.  Calo Stage1:  32 bits: if the first eight bits are 0xff, will read the "
        "74x MC format.\n");
    desc.add<unsigned int>("DmxFWId", 0)
        ->setComment(
            "Ignored unless FWOverride is true.  Calo Stage1:  32 bits: if the first eight bits are 0xff, will read "
            "the 74x MC format.\n");
    desc.add<bool>("FWOverride", false)->setComment("Firmware version should be taken as FWId parameters");
    desc.add<bool>("TMTCheck", true)->setComment("Flag for turning on/off Calo Layer 2 TMT node check");
    desc.addUntracked<bool>("CTP7", false);
    desc.addUntracked<bool>("MTF7", false);
    desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector"));
    desc.addUntracked<int>("lenSlinkHeader", 8);
    desc.addUntracked<int>("lenSlinkTrailer", 8);
    desc.addUntracked<int>("lenAMCHeader", 8);
    desc.addUntracked<int>("lenAMCTrailer", 0);
    desc.addUntracked<int>("lenAMC13Header", 8);
    desc.addUntracked<int>("lenAMC13Trailer", 8);
    desc.addUntracked<bool>("debug", false)->setComment("turn on verbose output");
    desc.add<unsigned int>("MinFeds", 0)
        ->setComment("optional parameter:  warn if less than MinFeds non-empty FED ids unpacked.");
    descriptions.add("l1tRawToDigi", desc);
  }
}  // namespace l1t

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(L1TRawToDigi);

#include "DQM/L1TMonitor/interface/L1TMP7ZeroSupp.h"

#include <sstream>


L1TMP7ZeroSupp::L1TMP7ZeroSupp(const edm::ParameterSet& ps)
    : fedDataToken_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("rawData"))),
      fedIds_(ps.getParameter<std::vector<int>>("fedIds")),
      slinkHeaderSize_(ps.getUntrackedParameter<int>("lenSlinkHeader", 8)),
      slinkTrailerSize_(ps.getUntrackedParameter<int>("lenSlinkTrailer", 8)),
      amcHeaderSize_(ps.getUntrackedParameter<int>("lenAMCHeader", 8)),
      amcTrailerSize_(ps.getUntrackedParameter<int>("lenAMCTrailer", 0)),
      amc13HeaderSize_(ps.getUntrackedParameter<int>("lenAMC13Header", 8)),
      amc13TrailerSize_(ps.getUntrackedParameter<int>("lenAMC13Trailer", 8)),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir", "")),
      verbose_(ps.getUntrackedParameter<bool>("verbose", false))
{
  std::vector<int> zeroMask(6, 0); 
  masks_.reserve(12);
  std::stringstream ss;
  for (size_t i = 0; i < 12; ++i) {
    ss.str("maskCapId");
    ss << i;
    masks_.push_back(ps.getUntrackedParameter<std::vector<int>>(ss.str().c_str(), zeroMask));
  }
}

L1TMP7ZeroSupp::~L1TMP7ZeroSupp() {}

void L1TMP7ZeroSupp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TMP7ZeroSupp::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TMP7ZeroSupp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {

  // Subsystem Monitoring and Muon Output
  ibooker.setCurrentFolder(monitorDir_);

  zeroSuppVal_ = ibooker.book1D("zeroSuppVal", "Zero suppression validation summary", 6, 0, 6);
  zeroSuppVal_->setAxisTitle("Block status", 1);
  zeroSuppVal_->setBinLabel(EVTS, "evts", 1);
  zeroSuppVal_->setBinLabel(BLOCKS, "blocks", 1);
  zeroSuppVal_->setBinLabel(ZSBLKSGOOD, "good", 1);
  zeroSuppVal_->setBinLabel(ZSBLKSBAD, "bad", 1);
  zeroSuppVal_->setBinLabel(ZSBLKSBADFALSEPOS, "false pos.", 1);
  zeroSuppVal_->setBinLabel(ZSBLKSBADFALSENEG, "false neg.", 1);
}

void L1TMP7ZeroSupp::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose_) edm::LogInfo("L1TMP7ZeroSupp") << "L1TMP7ZeroSupp: analyze..." << std::endl;

  edm::Handle<FEDRawDataCollection> feds;
  e.getByToken(fedDataToken_, feds);

  if (!feds.isValid()) {
    edm::LogError("L1TDQM") << "Cannot analyse: no FEDRawDataCollection found";
    return;
  }

  unsigned valid_count = 0;
  for (const auto& fedId: fedIds_) {
    unsigned int payloadSize = 0;
    unsigned int payloadSizeZS = 0;

    const FEDRawData& l1tRcd = feds->FEDData(fedId);

    edm::LogInfo("L1TDQM") << "Found FEDRawDataCollection with ID " << fedId << " and size " << l1tRcd.size();

    if ((int) l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ + amcHeaderSize_ + amcTrailerSize_) {
      if (l1tRcd.size() > 0) {
        edm::LogError("L1TDQM") << "Cannot analyse: invalid L1T raw data (size = " << l1tRcd.size() << ") for ID " << fedId << ".";
      }
      continue;
    } else {
      valid_count++;
    }

    const unsigned char *data = l1tRcd.data();
    FEDHeader header(data);

    if (header.check()) {
      edm::LogInfo("L1TDQM") << "Found SLink header:" << " Trigger type " << header.triggerType() << " L1 event ID " << header.lvl1ID() << " BX Number " << header.bxID() << " FED source " << header.sourceID() << " FED version " << header.version();
    } else {
      edm::LogWarning("L1TDQM") << "Did not find a SLink header!";
    }

    FEDTrailer trailer(data + (l1tRcd.size() - slinkTrailerSize_));

    if (trailer.check()) {
      edm::LogInfo("L1TDQM") << "Found SLink trailer:" << " Length " << trailer.lenght() << " CRC " << trailer.crc() << " Status " << trailer.evtStatus() << " Throttling bits " << trailer.ttsBits();
    } else {
      edm::LogWarning("L1TDQM") << "Did not find a SLink trailer!";
    }

    //amc13::Packet packet;
    //if (!packet.parse(
    //         (const uint64_t*) data,
    //         (const uint64_t*) (data + slinkHeaderSize_),
    //         (l1tRcd.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8,
    //         header.lvl1ID(),
    //         header.bxID(),
    //         false,
    //         false)) {
    //   edm::LogError("L1TDQM") << "Could not extract AMC13 Packet.";
    //   return;
    //}

    //for (auto& amc: packet.payload()) {
    //  if (amc.size() == 0)
    //    continue;

    //  auto payload64 = amc.data();
    //  const uint32_t * start = (const uint32_t*) payload64.get();
    //  // Want to have payload size in 32 bit words, but AMC measures
    //  // it in 64 bit words -> factor 2.
    //  const uint32_t * end = start + (amc.size() * 2);

    //  std::auto_ptr<l1t::Payload> payload;
    //  payload.reset(new l1t::MP7Payload(start, end, false));

    //  unsigned board = amc.blockHeader().getBoardID();
    //  unsigned amc_no = amc.blockHeader().getAMCNumber();

    //  bool zsFlagSet = false;
    //  bool toSuppress = false;
    //  // getBlock() returns a non-null auto_ptr on success
    //  std::auto_ptr<l1t::Block> block;
    //  while ((block = payload->getBlock()).get()) {
    //    if (verbose_) {
    //      std::cout << ">>> block to unpack <<<" << std::endl
    //                << "hdr:  " << std::hex << std::setw(8) << std::setfill('0') << block->header().raw() << std::dec
    //                << " (ID " << block->header().getID() << ", size " << block->header().getSize()
    //                << ", CapID 0x" << std::hex << std::setw(2) << std::setfill('0') << block->header().getCapID()
    //  	            << ")" << std::dec << std::endl;
    //      for (const auto& word: block->payload()) {
    //        if (verbose_) std::cout << "data: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
    //      }
    //    }

    //    unsigned int blockId = block->header().getID();
    //    unsigned int blockCapId = block->header().getCapID();
    //    unsigned int blocksize = block->header().getSize();

    //    // check the data

//  //      auto unpacker = unpackers.find(block->header().getID());
//
//  //      block->amc(amc.header());
//
//  //      if (unpacker == unpackers.end()) {
//  //        edm::LogInfo("L1TDQM") << "Cannot find an unpacker for"
//  //                        << "\n\tblock: ID " << block->header().getID() << ", size " << block->header().getSize()
//  //                        << "\n\tAMC: # " << amc_no << ", board ID 0x" << std::hex << board << std::dec
//  //                        << "\n\tFED ID " << fedId FW ID " << fw;
//  //      } else if (!unpacker->second->unpack(*block, coll.get())) {
//  //        edm::LogInfo("L1TDQM") << "Error unpacking data for block ID "
//  //                        << block->header().getID() << ", AMC # " << amc_no
//  //                        << ", board ID " << board << ", FED ID " << fedId
//  //                        << ", and FW ID " << fw << "!";
//  //      }
    //  }
    //}
  }
}


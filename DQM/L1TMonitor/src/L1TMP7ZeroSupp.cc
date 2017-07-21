#include "DQM/L1TMonitor/interface/L1TMP7ZeroSupp.h"

#include <sstream>

const unsigned int L1TMP7ZeroSupp::maxMasks_ = 16;

L1TMP7ZeroSupp::L1TMP7ZeroSupp(const edm::ParameterSet& ps)
    : fedDataToken_(consumes<FEDRawDataCollection>(ps.getParameter<edm::InputTag>("rawData"))),
      zsEnabled_(ps.getUntrackedParameter<bool>("zsEnabled")),
      fedIds_(ps.getParameter<std::vector<int>>("fedIds")),
      slinkHeaderSize_(ps.getUntrackedParameter<int>("lenSlinkHeader")),
      slinkTrailerSize_(ps.getUntrackedParameter<int>("lenSlinkTrailer")),
      amc13HeaderSize_(ps.getUntrackedParameter<int>("lenAMC13Header")),
      amc13TrailerSize_(ps.getUntrackedParameter<int>("lenAMC13Trailer")),
      amcHeaderSize_(ps.getUntrackedParameter<int>("lenAMCHeader")),
      amcTrailerSize_(ps.getUntrackedParameter<int>("lenAMCTrailer")),
      zsFlagMask_(ps.getUntrackedParameter<int>("zsFlagMask")),
      maxFedReadoutSize_(ps.getUntrackedParameter<int>("maxFEDReadoutSize")),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")),
      verbose_(ps.getUntrackedParameter<bool>("verbose"))
{
  std::vector<int> zeroMask(6, 0);
  masks_.reserve(maxMasks_);
  for (unsigned int i = 0; i < maxMasks_; ++i) {
    std::string maskCapIdStr{"maskCapId"+std::to_string(i)};
    masks_.push_back(ps.getUntrackedParameter<std::vector<int>>(maskCapIdStr.c_str(), zeroMask));
    // which masks are defined?
    if (ps.exists(maskCapIdStr.c_str())) {
      definedMaskCapIds_.push_back(i);
    }
  }
  if (verbose_) {
    // check masks
    std::cout << "masks" << std::endl;
    for (unsigned int i = 0; i < maxMasks_; ++i) {
      std::cout << "caption ID" << i << ":" << std::endl;
      for (const auto& maskIt: masks_.at(i)) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << maskIt << std::dec << std::endl;
      }
    }
    std::cout << "----------" << std::endl;
  }
}

L1TMP7ZeroSupp::~L1TMP7ZeroSupp() {}

void L1TMP7ZeroSupp::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("rawData");
  desc.add<std::vector<int>>("fedIds")->setComment("FED ids to analyze.");
  desc.addUntracked<bool>("zsEnabled", true)->setComment("MP7 zero suppression is enabled.");
  desc.addUntracked<int>("lenSlinkHeader", 8)->setComment("Number of Slink header bytes.");
  desc.addUntracked<int>("lenSlinkTrailer", 8)->setComment("Number of Slink trailer bytes.");
  desc.addUntracked<int>("lenAMC13Header", 8)->setComment("Number of AMC13 header bytes.");
  desc.addUntracked<int>("lenAMC13Trailer", 8)->setComment("Number of AMC13 trailer bytes.");
  desc.addUntracked<int>("lenAMCHeader", 8)->setComment("Number of AMC header bytes.");
  desc.addUntracked<int>("lenAMCTrailer", 0)->setComment("Number of AMC trailer bytes.");
  desc.addUntracked<int>("zsFlagMask", 0x1)->setComment("Zero suppression flag mask.");
  desc.addUntracked<int>("maxFEDReadoutSize", 10000)->setComment("Maximal FED readout size histogram x-axis value.");
  for (unsigned int i = 0; i < maxMasks_; ++i) {
    desc.addOptionalUntracked<std::vector<int>>(("maskCapId"+std::to_string(i)).c_str())->setComment(("ZS mask for caption id "+std::to_string(i)+".").c_str());
  }
  desc.addUntracked<std::string>("monitorDir", "")->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tMP7ZeroSupp", desc);
}

void L1TMP7ZeroSupp::dqmBeginRun(const edm::Run& r, const edm::EventSetup& c) {}

void L1TMP7ZeroSupp::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

void L1TMP7ZeroSupp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  // overall summary
  ibooker.setCurrentFolder(monitorDir_);
  bookCapIdHistograms(ibooker, maxMasks_);
  capIds_ = ibooker.book1D("capIds", "Caption ids found in data", maxMasks_, 0, maxMasks_);
  capIds_->setAxisTitle("caption id", 1);

  // per caption id subdirectories
  std::stringstream ss;
  for (const auto &id: definedMaskCapIds_) {
    ss.str("");
    ss << monitorDir_ << "/CapId" << id;
    ibooker.setCurrentFolder(ss.str().c_str());
    bookCapIdHistograms(ibooker, id);
  }
}

void L1TMP7ZeroSupp::bookCapIdHistograms(DQMStore::IBooker& ibooker, const unsigned int& id) {
  std::string summaryTitleText = "Zero suppression validation summary";
  std::string sizeTitleText;
  if (id == maxMasks_) {
    sizeTitleText = "FED readout ";
  } else {
    std::stringstream ss;
    ss << summaryTitleText << ", caption id " << id;
    summaryTitleText = ss.str();
    ss.str("");
    ss << "cumulated caption id " << id << " block ";
    sizeTitleText = ss.str();
  }

  zeroSuppValMap_[id] = ibooker.book1D("zeroSuppVal", summaryTitleText.c_str(), NBINLABELS, 0, NBINLABELS);
  zeroSuppValMap_[id]->setAxisTitle("Block status", 1);
  zeroSuppValMap_[id]->setBinLabel(EVTS+1, "events", 1);
  zeroSuppValMap_[id]->setBinLabel(EVTSGOOD+1, "good events", 1);
  zeroSuppValMap_[id]->setBinLabel(EVTSBAD+1, "bad events", 1);
  zeroSuppValMap_[id]->setBinLabel(BLOCKS+1, "blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSGOOD+1, "good blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSBAD+1, "bad blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSBADFALSEPOS+1, "false pos.", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSBADFALSENEG+1, "false neg.", 1);

  errorSummaryNumMap_[id] = ibooker.book1D("errorSummaryNum", summaryTitleText.c_str(), RNBINLABELS, 0, RNBINLABELS);
  errorSummaryNumMap_[id]->setBinLabel(REVTS+1, "bad events", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBLKS+1, "bad blocks", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBLKSFALSEPOS+1, "false pos.", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBLKSFALSENEG+1, "false neg.", 1);

  errorSummaryDenMap_[id] = ibooker.book1D("errorSummaryDen", "denominators", RNBINLABELS, 0, RNBINLABELS);
  errorSummaryDenMap_[id]->setBinLabel(REVTS+1, "# events", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBLKS+1, "# blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBLKSFALSEPOS+1, "# blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBLKSFALSENEG+1, "# blocks", 1);

  readoutSizeNoZSMap_[id] = ibooker.book1D("readoutSize", (sizeTitleText + "size").c_str(), 100, 0, maxFedReadoutSize_);
  readoutSizeNoZSMap_[id]->setAxisTitle("size (byte)", 1);
  readoutSizeZSMap_[id] = ibooker.book1D("readoutSizeZS", (sizeTitleText + "size with zero suppression").c_str(), 100, 0, maxFedReadoutSize_);
  readoutSizeZSMap_[id]->setAxisTitle("size (byte)", 1);
  readoutSizeZSExpectedMap_[id] = ibooker.book1D("readoutSizeZSExpected", ("Expected " + sizeTitleText + "size with zero suppression").c_str(), 100, 0, maxFedReadoutSize_);
  readoutSizeZSExpectedMap_[id]->setAxisTitle("size (byte)", 1);
}

void L1TMP7ZeroSupp::analyze(const edm::Event& e, const edm::EventSetup& c) {

  if (verbose_) edm::LogInfo("L1TDQM") << "L1TMP7ZeroSupp: analyze..." << std::endl;

  edm::Handle<FEDRawDataCollection> feds;
  e.getByToken(fedDataToken_, feds);

  if (!feds.isValid()) {
    edm::LogError("L1TDQM") << "Cannot analyse: no FEDRawDataCollection found";
    return;
  }

  zeroSuppValMap_[maxMasks_]->Fill(EVTS);
  errorSummaryDenMap_[maxMasks_]->Fill(REVTS);
  for (const auto &id: definedMaskCapIds_) {
    zeroSuppValMap_[id]->Fill(EVTS);
    errorSummaryDenMap_[id]->Fill(REVTS);
  }

  std::map<unsigned int, bool> evtGood;
  evtGood[maxMasks_] = true;
  for (const auto &id: definedMaskCapIds_) {
    evtGood[id] = true;
  }
  unsigned valid_count = 0;
  for (const auto& fedId: fedIds_) {
    const FEDRawData& l1tRcd = feds->FEDData(fedId);

    unsigned int fedDataSize = l1tRcd.size();
    std::map<unsigned int, unsigned int> readoutSizeNoZSMap;
    std::map<unsigned int, unsigned int> readoutSizeZSMap;
    std::map<unsigned int, unsigned int> readoutSizeZSExpectedMap;
    readoutSizeNoZSMap[maxMasks_] = 0;
    readoutSizeZSMap[maxMasks_] = 0;
    readoutSizeZSExpectedMap[maxMasks_] = 0;
    for (const auto &id: definedMaskCapIds_) {
      readoutSizeNoZSMap[id] = 0;
      readoutSizeZSMap[id] = 0;
      readoutSizeZSExpectedMap[id] = 0;
    }

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

    amc13::Packet packet;
    if (!packet.parse(
             (const uint64_t*) data,
             (const uint64_t*) (data + slinkHeaderSize_),
             (l1tRcd.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8,
             header.lvl1ID(),
             header.bxID())) {
       edm::LogError("L1TDQM") << "Could not extract AMC13 Packet.";
       return;
    }

    for (auto& amc: packet.payload()) {
      if (amc.size() == 0)
        continue;

      auto payload64 = amc.data();
      const uint32_t * start = (const uint32_t*) payload64.get();
      // Want to have payload size in 32 bit words, but AMC measures
      // it in 64 bit words -> factor 2.
      const uint32_t * end = start + (amc.size() * 2);

      std::unique_ptr<l1t::Payload> payload;
      payload.reset(new l1t::MP7Payload(start, end, false));

      // getBlock() returns a non-null unique_ptr on success
      std::unique_ptr<l1t::Block> block;
      while ((block = payload->getBlock()).get()) {
        if (verbose_) {
          std::cout << ">>> check zero suppression for block <<<" << std::endl
                    << "hdr:  " << std::hex << std::setw(8) << std::setfill('0') << block->header().raw() << std::dec
                    << " (ID " << block->header().getID() << ", size " << block->header().getSize()
                    << ", CapID 0x" << std::hex << std::setw(2) << std::setfill('0') << block->header().getCapID()
                    << ", flags 0x" << std::hex << std::setw(2) << std::setfill('0') << block->header().getFlags()
                    << ")" << std::dec << std::endl;
          for (const auto& word: block->payload()) {
            std::cout << "data: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
          }
        }

        unsigned int blockCapId = block->header().getCapID();
        unsigned int blockSize = block->header().getSize() * 4;
        unsigned int blockHeaderSize = sizeof(block->header().raw());
        bool zsFlagSet = ((block->header().getFlags() & zsFlagMask_) != 0);
        bool toSuppress = false;

        capIds_->Fill(blockCapId);

        bool capIdDefined = false;
        zeroSuppValMap_[maxMasks_]->Fill(BLOCKS);
        errorSummaryDenMap_[maxMasks_]->Fill(RBLKS);
        errorSummaryDenMap_[maxMasks_]->Fill(RBLKSFALSEPOS);
        errorSummaryDenMap_[maxMasks_]->Fill(RBLKSFALSENEG);
        if (zeroSuppValMap_.find(blockCapId) != zeroSuppValMap_.end()) {
          capIdDefined = true;
          zeroSuppValMap_[blockCapId]->Fill(BLOCKS);
          errorSummaryDenMap_[blockCapId]->Fill(RBLKS);
          errorSummaryDenMap_[blockCapId]->Fill(RBLKSFALSEPOS);
          errorSummaryDenMap_[blockCapId]->Fill(RBLKSFALSENEG);
        }

        auto totalBlockSize = blockSize + blockHeaderSize;
        readoutSizeNoZSMap[maxMasks_] += totalBlockSize;
        if (capIdDefined) {
          readoutSizeNoZSMap[blockCapId] += totalBlockSize;
        }

        // check if this block should be suppressed
        unsigned int wordcounter = 0;
        unsigned int wordsum = 0;
        for (const auto& word: block->payload()) {
          wordsum += masks_[blockCapId].at(wordcounter%6) & word;
          if (verbose_) {
            std::cout << "word: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec
                      << ", maskword" << wordcounter%6 << ": " << std::hex << std::setw(8) << std::setfill('0')
                      << masks_[blockCapId].at(wordcounter%6) << std::dec << ", wordsum: " << wordsum << std::endl;
          }
          if (wordsum > 0) {
            break;
          }
          ++wordcounter;
        }
        if (wordsum == 0 && zsEnabled_) {
          toSuppress = true;
          if (verbose_) {
            std::cout << "wordsum == 0: this block should be zero suppressed" << std::endl;
          }
        }

        // check if zero suppression flag agrees
        if (toSuppress && zsFlagSet) {
          if (verbose_) std::cout << "GOOD block with ZS flag true" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSGOOD);
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSGOOD);
          }
        } else if (!toSuppress && !zsFlagSet) {
          if (verbose_) std::cout << "GOOD block with ZS flag false" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSGOOD);
          readoutSizeZSMap[maxMasks_] += totalBlockSize;
          readoutSizeZSExpectedMap[maxMasks_] += totalBlockSize;
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSGOOD);
            readoutSizeZSMap[blockCapId] += totalBlockSize;
            readoutSizeZSExpectedMap[blockCapId] += totalBlockSize;
          }
        } else if (!toSuppress && zsFlagSet) {
          if (verbose_) std::cout << "BAD block with ZS flag true" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSBAD);
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSBADFALSEPOS);
          errorSummaryNumMap_[maxMasks_]->Fill(RBLKS);
          errorSummaryNumMap_[maxMasks_]->Fill(RBLKSFALSEPOS);
          readoutSizeZSExpectedMap[maxMasks_] += totalBlockSize;
          evtGood[maxMasks_] = false;
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSBAD);
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSBADFALSEPOS);
            errorSummaryNumMap_[blockCapId]->Fill(RBLKS);
            errorSummaryNumMap_[blockCapId]->Fill(RBLKSFALSEPOS);
            readoutSizeZSExpectedMap[blockCapId] += totalBlockSize;
            evtGood[blockCapId] = false;
          }
        } else {
          if (verbose_) std::cout << "BAD block with ZS flag false" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSBAD);
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSBADFALSENEG);
          errorSummaryNumMap_[maxMasks_]->Fill(RBLKS);
          errorSummaryNumMap_[maxMasks_]->Fill(RBLKSFALSENEG);
          readoutSizeZSMap[maxMasks_] += totalBlockSize;
          evtGood[maxMasks_] = false;
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSBAD);
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSBADFALSENEG);
            errorSummaryNumMap_[blockCapId]->Fill(RBLKS);
            errorSummaryNumMap_[blockCapId]->Fill(RBLKSFALSENEG);
            readoutSizeZSMap[blockCapId] += totalBlockSize;
            evtGood[blockCapId] = false;
          }
        }
      }
    }
    readoutSizeNoZSMap_[maxMasks_]->Fill(fedDataSize);
    readoutSizeZSMap_[maxMasks_]->Fill(readoutSizeZSMap[maxMasks_] + fedDataSize - readoutSizeNoZSMap[maxMasks_]);
    readoutSizeZSExpectedMap_[maxMasks_]->Fill(readoutSizeZSExpectedMap[maxMasks_] + fedDataSize - readoutSizeNoZSMap[maxMasks_]);
    for (const auto &id: definedMaskCapIds_) {
      readoutSizeNoZSMap_[id]->Fill(readoutSizeNoZSMap[id]);
      readoutSizeZSMap_[id]->Fill(readoutSizeZSMap[id]);
      readoutSizeZSExpectedMap_[id]->Fill(readoutSizeZSExpectedMap[id]);
    }
  }

  if (evtGood[maxMasks_]) {
    zeroSuppValMap_[maxMasks_]->Fill(EVTSGOOD);
  } else {
    zeroSuppValMap_[maxMasks_]->Fill(EVTSBAD);
    errorSummaryNumMap_[maxMasks_]->Fill(REVTS);
  }
  for (const auto &id: definedMaskCapIds_) {
    if (evtGood[id]) {
      zeroSuppValMap_[id]->Fill(EVTSGOOD);
    } else {
      zeroSuppValMap_[id]->Fill(EVTSBAD);
      errorSummaryNumMap_[id]->Fill(REVTS);
    }
  }
}


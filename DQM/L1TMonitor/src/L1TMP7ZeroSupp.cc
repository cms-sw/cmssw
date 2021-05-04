#include "DQM/L1TMonitor/interface/L1TMP7ZeroSupp.h"

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
      newZsFlagMask_(ps.getUntrackedParameter<int>("newZsFlagMask")),
      zsFlagMask_(ps.getUntrackedParameter<int>("zsFlagMask")),
      dataInvFlagMask_(ps.getUntrackedParameter<int>("dataInvFlagMask")),
      maxFedReadoutSize_(ps.getUntrackedParameter<int>("maxFEDReadoutSize")),
      checkOnlyCapIdsWithMasks_(ps.getUntrackedParameter<bool>("checkOnlyCapIdsWithMasks")),
      monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir")),
      verbose_(ps.getUntrackedParameter<bool>("verbose")) {
  std::vector<int> onesMask(6, 0xffffffff);
  masks_.reserve(maxMasks_);
  for (unsigned int i = 0; i < maxMasks_; ++i) {
    std::string maskCapIdStr{"maskCapId" + std::to_string(i)};
    masks_.push_back(ps.getUntrackedParameter<std::vector<int>>(maskCapIdStr, onesMask));
    // which masks are defined?
    if (ps.exists(maskCapIdStr)) {
      definedMaskCapIds_.push_back(i);
    }
  }
  if (verbose_) {
    // check masks
    std::cout << "masks" << std::endl;
    for (unsigned int i = 0; i < maxMasks_; ++i) {
      std::cout << "caption ID" << i << ":" << std::endl;
      for (const auto& maskIt : masks_.at(i)) {
        std::cout << std::hex << std::setw(8) << std::setfill('0') << maskIt << std::dec << std::endl;
      }
    }
    std::cout << "----------" << std::endl;
  }
}

L1TMP7ZeroSupp::~L1TMP7ZeroSupp() = default;

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
  desc.addUntracked<int>("newZsFlagMask", 0x2)->setComment("Per-BX zero suppression flag mask.");
  desc.addUntracked<int>("dataInvFlagMask", 0x1)->setComment("Data inversion flag mask.");
  desc.addUntracked<int>("maxFEDReadoutSize", 10000)->setComment("Maximal FED readout size histogram x-axis value.");
  for (unsigned int i = 0; i < maxMasks_; ++i) {
    desc.addOptionalUntracked<std::vector<int>>("maskCapId" + std::to_string(i))
        ->setComment("ZS mask for caption id " + std::to_string(i) + ".");
  }
  desc.addUntracked<bool>("checkOnlyCapIdsWithMasks", true)
      ->setComment("Check only blocks that have a CapId for which a mask is defined.");
  desc.addUntracked<std::string>("monitorDir", "")
      ->setComment("Target directory in the DQM file. Will be created if not existing.");
  desc.addUntracked<bool>("verbose", false);
  descriptions.add("l1tMP7ZeroSupp", desc);
}

void L1TMP7ZeroSupp::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) {
  // overall summary
  ibooker.setCurrentFolder(monitorDir_);
  bookCapIdHistograms(ibooker, maxMasks_);
  capIds_ = ibooker.book1D("capIds", "Caption ids found in data", maxMasks_, 0, maxMasks_);
  capIds_->setAxisTitle("caption id", 1);

  // per caption id subdirectories
  for (const auto& id : definedMaskCapIds_) {
    ibooker.setCurrentFolder(monitorDir_ + "/CapId" + std::to_string(id));
    bookCapIdHistograms(ibooker, id);
  }
}

void L1TMP7ZeroSupp::bookCapIdHistograms(DQMStore::IBooker& ibooker, const unsigned int& id) {
  std::string summaryTitleText = "Zero suppression validation summary";
  std::string sizeTitleText;
  if (id == maxMasks_) {
    sizeTitleText = "FED readout ";
  } else {
    summaryTitleText = summaryTitleText + ", caption id " + std::to_string(id);
    sizeTitleText = "cumulated caption id " + std::to_string(id) + " block ";
  }

  zeroSuppValMap_[id] = ibooker.book1D("zeroSuppVal", summaryTitleText, (int)NBINLABELS, 0, (int)NBINLABELS);
  zeroSuppValMap_[id]->setAxisTitle("ZS status", 1);
  zeroSuppValMap_[id]->setBinLabel(EVTS + 1, "events", 1);
  zeroSuppValMap_[id]->setBinLabel(EVTSGOOD + 1, "good events", 1);
  zeroSuppValMap_[id]->setBinLabel(EVTSBAD + 1, "bad events", 1);
  zeroSuppValMap_[id]->setBinLabel(BLOCKS + 1, "blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSGOOD + 1, "good blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSBAD + 1, "bad blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSBADFALSEPOS + 1, "false pos.", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBLKSBADFALSENEG + 1, "false neg.", 1);
  zeroSuppValMap_[id]->setBinLabel(BXBLOCKS + 1, "BX blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBXBLKSGOOD + 1, "good BX blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBXBLKSBAD + 1, "bad BX blocks", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBXBLKSBADFALSEPOS + 1, "BX false pos.", 1);
  zeroSuppValMap_[id]->setBinLabel(ZSBXBLKSBADFALSENEG + 1, "BX false neg.", 1);

  errorSummaryNumMap_[id] = ibooker.book1D("errorSummaryNum", summaryTitleText, (int)RNBINLABELS, 0, (int)RNBINLABELS);
  errorSummaryNumMap_[id]->setBinLabel(REVTS + 1, "bad events", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBLKS + 1, "bad blocks", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBLKSFALSEPOS + 1, "false pos.", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBLKSFALSENEG + 1, "false neg.", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBXBLKS + 1, "bad BX blocks", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBXBLKSFALSEPOS + 1, "BX false pos.", 1);
  errorSummaryNumMap_[id]->setBinLabel(RBXBLKSFALSENEG + 1, "BX false neg.", 1);

  errorSummaryDenMap_[id] = ibooker.book1D("errorSummaryDen", "denominators", (int)RNBINLABELS, 0, (int)RNBINLABELS);
  errorSummaryDenMap_[id]->setBinLabel(REVTS + 1, "# events", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBLKS + 1, "# blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBLKSFALSEPOS + 1, "# blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBLKSFALSENEG + 1, "# blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBXBLKS + 1, "# BX blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBXBLKSFALSEPOS + 1, "# BX blocks", 1);
  errorSummaryDenMap_[id]->setBinLabel(RBXBLKSFALSENEG + 1, "# BX blocks", 1);
  // Setting canExtend to false is needed to get the correct behaviour when running multithreaded.
  // Otherwise, when merging the histgrams of the threads, TH1::Merge sums bins that have the same label in one bin.
  // This needs to come after the calls to setBinLabel.
  errorSummaryDenMap_[id]->getTH1F()->GetXaxis()->SetCanExtend(false);

  readoutSizeNoZSMap_[id] = ibooker.book1D("readoutSize", sizeTitleText + "size", 100, 0, maxFedReadoutSize_);
  readoutSizeNoZSMap_[id]->setAxisTitle("size (byte)", 1);
  readoutSizeZSMap_[id] =
      ibooker.book1D("readoutSizeZS", sizeTitleText + "size with zero suppression", 100, 0, maxFedReadoutSize_);
  readoutSizeZSMap_[id]->setAxisTitle("size (byte)", 1);
  readoutSizeZSExpectedMap_[id] = ibooker.book1D(
      "readoutSizeZSExpected", "Expected " + sizeTitleText + "size with zero suppression", 100, 0, maxFedReadoutSize_);
  readoutSizeZSExpectedMap_[id]->setAxisTitle("size (byte)", 1);
}

void L1TMP7ZeroSupp::analyze(const edm::Event& e, const edm::EventSetup& c) {
  if (verbose_)
    edm::LogInfo("L1TDQM") << "L1TMP7ZeroSupp: analyze..." << std::endl;

  edm::Handle<FEDRawDataCollection> feds;
  e.getByToken(fedDataToken_, feds);

  if (!feds.isValid()) {
    edm::LogError("L1TDQM") << "Cannot analyse: no FEDRawDataCollection found";
    return;
  }

  zeroSuppValMap_[maxMasks_]->Fill(EVTS);
  errorSummaryDenMap_[maxMasks_]->Fill(REVTS);
  for (const auto& id : definedMaskCapIds_) {
    zeroSuppValMap_[id]->Fill(EVTS);
    errorSummaryDenMap_[id]->Fill(REVTS);
  }

  std::map<unsigned int, bool> evtGood;
  evtGood[maxMasks_] = true;
  for (const auto& id : definedMaskCapIds_) {
    evtGood[id] = true;
  }
  unsigned valid_count = 0;
  for (const auto& fedId : fedIds_) {
    const FEDRawData& l1tRcd = feds->FEDData(fedId);

    unsigned int fedDataSize = l1tRcd.size();
    std::map<unsigned int, unsigned int> readoutSizeNoZSMap;
    std::map<unsigned int, unsigned int> readoutSizeZSMap;
    std::map<unsigned int, unsigned int> readoutSizeZSExpectedMap;
    readoutSizeNoZSMap[maxMasks_] = 0;
    readoutSizeZSMap[maxMasks_] = 0;
    readoutSizeZSExpectedMap[maxMasks_] = 0;
    for (const auto& id : definedMaskCapIds_) {
      readoutSizeNoZSMap[id] = 0;
      readoutSizeZSMap[id] = 0;
      readoutSizeZSExpectedMap[id] = 0;
    }

    edm::LogInfo("L1TDQM") << "Found FEDRawDataCollection with ID " << fedId << " and size " << l1tRcd.size();

    if ((int)l1tRcd.size() < slinkHeaderSize_ + slinkTrailerSize_ + amc13HeaderSize_ + amc13TrailerSize_ +
                                 amcHeaderSize_ + amcTrailerSize_) {
      if (l1tRcd.size() > 0) {
        edm::LogError("L1TDQM") << "Cannot analyse: invalid L1T raw data (size = " << l1tRcd.size() << ") for ID "
                                << fedId << ".";
      }
      continue;
    } else {
      valid_count++;
    }

    const unsigned char* data = l1tRcd.data();
    FEDHeader header(data);

    if (header.check()) {
      edm::LogInfo("L1TDQM") << "Found SLink header:"
                             << " Trigger type " << header.triggerType() << " L1 event ID " << header.lvl1ID()
                             << " BX Number " << header.bxID() << " FED source " << header.sourceID() << " FED version "
                             << header.version();
    } else {
      edm::LogWarning("L1TDQM") << "Did not find a SLink header!";
    }

    FEDTrailer trailer(data + (l1tRcd.size() - slinkTrailerSize_));

    if (trailer.check()) {
      edm::LogInfo("L1TDQM") << "Found SLink trailer:"
                             << " Length " << trailer.fragmentLength() << " CRC " << trailer.crc() << " Status "
                             << trailer.evtStatus() << " Throttling bits " << trailer.ttsBits();
    } else {
      edm::LogWarning("L1TDQM") << "Did not find a SLink trailer!";
    }

    amc13::Packet packet;
    if (!packet.parse((const uint64_t*)data,
                      (const uint64_t*)(data + slinkHeaderSize_),
                      (l1tRcd.size() - slinkHeaderSize_ - slinkTrailerSize_) / 8,
                      header.lvl1ID(),
                      header.bxID())) {
      edm::LogError("L1TDQM") << "Could not extract AMC13 Packet.";
      return;
    }

    for (auto& amc : packet.payload()) {
      if (amc.size() == 0)
        continue;

      auto payload64 = amc.data();
      auto start = (const uint32_t*)payload64.get();
      // Want to have payload size in 32 bit words, but AMC measures
      // it in 64 bit words -> factor 2.
      const uint32_t* end = start + (amc.size() * 2);

      auto payload = std::make_unique<l1t::MP7Payload>(start, end, false);

      // getBlock() returns a non-null unique_ptr on success
      std::unique_ptr<l1t::Block> block;
      while ((block = payload->getBlock()) != nullptr) {
        if (verbose_) {
          std::cout << ">>> check zero suppression for block <<<" << std::endl
                    << "hdr:  " << std::hex << std::setw(8) << std::setfill('0') << block->header().raw() << std::dec
                    << " (ID " << block->header().getID() << ", size " << block->header().getSize() << ", CapID 0x"
                    << std::hex << std::setw(2) << std::setfill('0') << block->header().getCapID() << ", flags 0x"
                    << std::hex << std::setw(2) << std::setfill('0') << block->header().getFlags() << ")" << std::dec
                    << std::endl;
          for (const auto& word : block->payload()) {
            std::cout << "data: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
          }
        }

        unsigned int blockCapId = block->header().getCapID();
        unsigned int blockSize = block->header().getSize() * 4;  // times 4 to get the size in byte
        unsigned int blockHeaderSize = sizeof(block->header().raw());
        unsigned int blockHeaderFlags = block->header().getFlags();
        bool newZsFlagSet = ((blockHeaderFlags & newZsFlagMask_) != 0);  // use the per-BX ZS
        bool blockZsFlagSet =
            newZsFlagSet ? true : ((blockHeaderFlags & zsFlagMask_) != 0);  // ZS validation flag for whole block
        bool dataInvertFlagSet =
            newZsFlagSet && ((blockHeaderFlags & dataInvFlagMask_) != 0);  // invert the data before applying the mask

        capIds_->Fill(blockCapId);

        bool capIdDefined = false;
        if (zeroSuppValMap_.find(blockCapId) != zeroSuppValMap_.end()) {
          capIdDefined = true;
        }

        // Only check blocks with a CapId that has a defined ZS mask.
        if (checkOnlyCapIdsWithMasks_ and not capIdDefined) {
          continue;
        }

        // fill the denominator histograms
        zeroSuppValMap_[maxMasks_]->Fill(BLOCKS);
        errorSummaryDenMap_[maxMasks_]->Fill(RBLKS);
        errorSummaryDenMap_[maxMasks_]->Fill(RBLKSFALSEPOS);
        errorSummaryDenMap_[maxMasks_]->Fill(RBLKSFALSENEG);
        if (capIdDefined) {
          zeroSuppValMap_[blockCapId]->Fill(BLOCKS);
          errorSummaryDenMap_[blockCapId]->Fill(RBLKS);
          errorSummaryDenMap_[blockCapId]->Fill(RBLKSFALSEPOS);
          errorSummaryDenMap_[blockCapId]->Fill(RBLKSFALSENEG);
        }

        auto totalBlockSize = blockHeaderSize;
        if (!newZsFlagSet) {
          totalBlockSize += blockSize;
        }
        auto totalBlockSizeExpected = totalBlockSize;
        auto totalBlockSizeNoZS = blockHeaderSize + blockSize;

        auto bxBlocks = block->getBxBlocks(6, newZsFlagSet);  // 6 32 bit MP7 payload words per BX

        // check all BX blocks
        bool allToSuppress = true;
        for (const auto& bxBlock : bxBlocks) {
          bool toSuppress = false;
          bool bxZsFlagSet = ((bxBlock.header().getFlags() & zsFlagMask_) != 0);  // ZS validation flag

          // check if this bxblock should be suppressed
          unsigned int wordcounter = 0;
          unsigned int wordsum = 0;
          for (const auto& word : bxBlock.payload()) {
            if (dataInvertFlagSet) {
              wordsum += masks_[blockCapId].at(wordcounter % 6) & (~word);
            } else {
              wordsum += masks_[blockCapId].at(wordcounter % 6) & word;
            }
            if (verbose_) {
              std::cout << "word: " << std::hex << std::setw(8) << std::setfill('0') << word << std::dec << ", maskword"
                        << wordcounter % 6 << ": " << std::hex << std::setw(8) << std::setfill('0')
                        << masks_[blockCapId].at(wordcounter % 6) << std::dec << ", wordsum: " << wordsum << std::endl;
            }
            if (wordsum > 0) {
              if (verbose_) {
                std::cout << "wordsum not 0: this BX block should be kept" << std::endl;
              }
              break;
            }
            ++wordcounter;
          }
          // the sum of payload words must be 0 for correct ZS
          if (wordsum == 0 && zsEnabled_) {
            toSuppress = true;
            if (verbose_) {
              std::cout << "wordsum == 0: this BX block should be zero suppressed" << std::endl;
            }
          }
          // update the overall block status
          allToSuppress = allToSuppress && toSuppress;

          // only fill the BX related things for the per-BX ZS
          if (newZsFlagSet) {
            // the ZS flag of the block is the AND of all BX block ZS flags
            blockZsFlagSet = blockZsFlagSet && bxZsFlagSet;

            // fill the BX related bins of the denominator histogram
            zeroSuppValMap_[maxMasks_]->Fill(BXBLOCKS);
            errorSummaryDenMap_[maxMasks_]->Fill(RBXBLKS);
            errorSummaryDenMap_[maxMasks_]->Fill(RBXBLKSFALSEPOS);
            errorSummaryDenMap_[maxMasks_]->Fill(RBXBLKSFALSENEG);
            if (capIdDefined) {
              zeroSuppValMap_[blockCapId]->Fill(BXBLOCKS);
              errorSummaryDenMap_[blockCapId]->Fill(RBXBLKS);
              errorSummaryDenMap_[blockCapId]->Fill(RBXBLKSFALSEPOS);
              errorSummaryDenMap_[blockCapId]->Fill(RBXBLKSFALSENEG);
            }

            unsigned int totalBxBlockSize =
                bxBlock.getSize() * 4 + sizeof(bxBlock.header().raw());  // times 4 to get the size in byte
            // check if zero suppression flag agrees for the BX block
            if (toSuppress && bxZsFlagSet) {
              if (verbose_)
                std::cout << "GOOD BX block with ZS flag true" << std::endl;
              zeroSuppValMap_[maxMasks_]->Fill(ZSBXBLKSGOOD);
              if (capIdDefined) {
                zeroSuppValMap_[blockCapId]->Fill(ZSBXBLKSGOOD);
              }
            } else if (!toSuppress && !bxZsFlagSet) {
              if (verbose_)
                std::cout << "GOOD BX block with ZS flag false" << std::endl;
              totalBlockSize += totalBxBlockSize;
              totalBlockSizeExpected += totalBxBlockSize;
              zeroSuppValMap_[maxMasks_]->Fill(ZSBXBLKSGOOD);
              if (capIdDefined) {
                zeroSuppValMap_[blockCapId]->Fill(ZSBXBLKSGOOD);
              }
            } else if (!toSuppress && bxZsFlagSet) {
              if (verbose_)
                std::cout << "BAD BX block with ZS flag true" << std::endl;
              totalBlockSizeExpected += totalBxBlockSize;
              zeroSuppValMap_[maxMasks_]->Fill(ZSBXBLKSBAD);
              zeroSuppValMap_[maxMasks_]->Fill(ZSBXBLKSBADFALSEPOS);
              errorSummaryNumMap_[maxMasks_]->Fill(RBXBLKS);
              errorSummaryNumMap_[maxMasks_]->Fill(RBXBLKSFALSEPOS);
              evtGood[maxMasks_] = false;
              if (capIdDefined) {
                zeroSuppValMap_[blockCapId]->Fill(ZSBXBLKSBAD);
                zeroSuppValMap_[blockCapId]->Fill(ZSBXBLKSBADFALSEPOS);
                errorSummaryNumMap_[blockCapId]->Fill(RBXBLKS);
                errorSummaryNumMap_[blockCapId]->Fill(RBXBLKSFALSEPOS);
                evtGood[blockCapId] = false;
              }
            } else {
              if (verbose_)
                std::cout << "BAD BX block with ZS flag false" << std::endl;
              totalBlockSize += totalBxBlockSize;
              zeroSuppValMap_[maxMasks_]->Fill(ZSBXBLKSBAD);
              zeroSuppValMap_[maxMasks_]->Fill(ZSBXBLKSBADFALSENEG);
              errorSummaryNumMap_[maxMasks_]->Fill(RBXBLKS);
              errorSummaryNumMap_[maxMasks_]->Fill(RBXBLKSFALSENEG);
              evtGood[maxMasks_] = false;
              if (capIdDefined) {
                zeroSuppValMap_[blockCapId]->Fill(ZSBXBLKSBAD);
                zeroSuppValMap_[blockCapId]->Fill(ZSBXBLKSBADFALSENEG);
                errorSummaryNumMap_[blockCapId]->Fill(RBXBLKS);
                errorSummaryNumMap_[blockCapId]->Fill(RBXBLKSFALSENEG);
                evtGood[blockCapId] = false;
              }
            }
          }
        }

        readoutSizeNoZSMap[maxMasks_] += totalBlockSizeNoZS;
        if (capIdDefined) {
          readoutSizeNoZSMap[blockCapId] += totalBlockSizeNoZS;
        }

        // check if zero suppression flag agrees for the whole block
        if (allToSuppress && blockZsFlagSet) {
          if (verbose_)
            std::cout << "GOOD block with ZS flag true" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSGOOD);
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSGOOD);
          }
        } else if (!allToSuppress && !blockZsFlagSet) {
          if (verbose_)
            std::cout << "GOOD block with ZS flag false" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSGOOD);
          readoutSizeZSMap[maxMasks_] += totalBlockSize;
          readoutSizeZSExpectedMap[maxMasks_] += totalBlockSizeExpected;
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSGOOD);
            readoutSizeZSMap[blockCapId] += totalBlockSize;
            readoutSizeZSExpectedMap[blockCapId] += totalBlockSizeExpected;
          }
        } else if (!allToSuppress && blockZsFlagSet) {
          if (verbose_)
            std::cout << "BAD block with ZS flag true" << std::endl;
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSBAD);
          zeroSuppValMap_[maxMasks_]->Fill(ZSBLKSBADFALSEPOS);
          errorSummaryNumMap_[maxMasks_]->Fill(RBLKS);
          errorSummaryNumMap_[maxMasks_]->Fill(RBLKSFALSEPOS);
          readoutSizeZSExpectedMap[maxMasks_] += totalBlockSizeExpected;
          evtGood[maxMasks_] = false;
          if (capIdDefined) {
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSBAD);
            zeroSuppValMap_[blockCapId]->Fill(ZSBLKSBADFALSEPOS);
            errorSummaryNumMap_[blockCapId]->Fill(RBLKS);
            errorSummaryNumMap_[blockCapId]->Fill(RBLKSFALSEPOS);
            readoutSizeZSExpectedMap[blockCapId] += totalBlockSizeExpected;
            evtGood[blockCapId] = false;
          }
        } else {
          if (verbose_)
            std::cout << "BAD block with ZS flag false" << std::endl;
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
    if (verbose_) {
      std::cout << "FED data size: " << fedDataSize << " bytes" << std::endl;
      std::cout << "Payload size no ZS: " << readoutSizeNoZSMap[maxMasks_] << " bytes" << std::endl;
      std::cout << "Payload size ZS: " << readoutSizeZSMap[maxMasks_] << " bytes" << std::endl;
      std::cout << "Payload size expected ZS: " << readoutSizeZSExpectedMap[maxMasks_] << " bytes" << std::endl;
      std::cout << "Filled readout size ZS with headers: "
                << readoutSizeZSMap[maxMasks_] + fedDataSize - readoutSizeNoZSMap[maxMasks_] << " bytes" << std::endl;
      std::cout << "Filled expected readout size ZS with headers: "
                << readoutSizeZSExpectedMap[maxMasks_] + fedDataSize - readoutSizeNoZSMap[maxMasks_] << " bytes"
                << std::endl;
    }
    readoutSizeNoZSMap_[maxMasks_]->Fill(fedDataSize);
    readoutSizeZSMap_[maxMasks_]->Fill(readoutSizeZSMap[maxMasks_] + fedDataSize - readoutSizeNoZSMap[maxMasks_]);
    readoutSizeZSExpectedMap_[maxMasks_]->Fill(readoutSizeZSExpectedMap[maxMasks_] + fedDataSize -
                                               readoutSizeNoZSMap[maxMasks_]);
    for (const auto& id : definedMaskCapIds_) {
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
  for (const auto& id : definedMaskCapIds_) {
    if (evtGood[id]) {
      zeroSuppValMap_[id]->Fill(EVTSGOOD);
    } else {
      zeroSuppValMap_[id]->Fill(EVTSBAD);
      errorSummaryNumMap_[id]->Fill(REVTS);
    }
  }
}

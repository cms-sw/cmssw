#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TCorrelatorLayer1PatternFileWriter.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

L1TCorrelatorLayer1PatternFileWriter::L1TCorrelatorLayer1PatternFileWriter(const edm::ParameterSet& iConfig,
                                                                           const l1ct::Event& eventTemplate)
    : partition_(parsePartition(iConfig.getParameter<std::string>("partition"))),
      writeInputs_(iConfig.existsAs<std::string>("inputFileName") &&
                   !iConfig.getParameter<std::string>("inputFileName").empty()),
      writeOutputs_(iConfig.existsAs<std::string>("outputFileName") &&
                    !iConfig.getParameter<std::string>("outputFileName").empty()),
      outputBoard_(-1),
      outputLinkEgamma_(-1),
      fileFormat_(iConfig.getParameter<std::string>("fileFormat")),
      eventsPerFile_(iConfig.getParameter<uint32_t>("eventsPerFile")),
      eventIndex_(0) {
  if (writeInputs_) {
    nInputFramesPerBX_ = iConfig.getParameter<uint32_t>("nInputFramesPerBX");

    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal) {
      configTimeSlices(iConfig, "tf", eventTemplate.raw.track.size(), tfTimeslices_, tfLinksFactor_);
      channelSpecsInput_["tf"] = {tmuxFactor_ * tfTimeslices_, tfTimeslices_};
    }
    if (partition_ == Partition::Barrel) {
      auto sectorConfig = iConfig.getParameter<std::vector<edm::ParameterSet>>("gctSectors");
      if (sectorConfig.size() != gctSectors_)
        throw cms::Exception("Configuration", "Bad number of GCT sectors");
      for (unsigned int iS = 0; iS < gctSectors_; ++iS) {
        auto linksEcal = sectorConfig[iS].getParameter<std::vector<int32_t>>("gctLinksEcal");
        auto linksHad = sectorConfig[iS].getParameter<std::vector<int32_t>>("gctLinksHad");
        if (linksEcal.size() != gctLinksEcal_ || linksHad.size() != gctLinksHad_)
          throw cms::Exception("Configuration", "Bad number of GCT links");
        unsigned int iLink = 0;
        for (unsigned int i = 0; i < gctLinksHad_; ++i, ++iLink) {
          if (linksHad[i] != -1)
            channelIdsInput_[l1t::demo::LinkId{"gct", iLink + 10 * iS}].push_back(linksHad[i]);
        }
        for (unsigned int i = 0; i < gctLinksEcal_; ++i) {
          if (linksEcal[i] != -1)
            channelIdsInput_[l1t::demo::LinkId{"gct", iLink + 10 * iS}].push_back(linksEcal[i]);
        }
        channelSpecsInput_["gct"] = {tmuxFactor_ * gctTimeslices_, 0};
      }
    }
    if (partition_ == Partition::HGCal || partition_ == Partition::HGCalNoTk) {
      configTimeSlices(iConfig, "hgc", eventTemplate.raw.hgcalcluster.size(), hgcTimeslices_, hgcLinksFactor_);
      channelSpecsInput_["hgc"] = {tmuxFactor_ * hgcTimeslices_, hgcTimeslices_};
    }
    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal || partition_ == Partition::HGCalNoTk) {
      configTimeSlices(iConfig, "gmt", 1, gmtTimeslices_, gmtLinksFactor_);
      gmtNumberOfMuons_ = iConfig.getParameter<uint32_t>("gmtNumberOfMuons");
      channelSpecsInput_["gmt"] = {tmuxFactor_ * gmtTimeslices_,
                                   gmtTimeslices_ * nInputFramesPerBX_ * tmuxFactor_ - gmtNumberOfMuons_};
    }
    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal) {
      configTimeSlices(iConfig, "gtt", 1, gttTimeslices_, gttLinksFactor_);
      gttLatency_ = iConfig.getParameter<uint32_t>("gttLatency");
      gttNumberOfPVs_ = iConfig.getParameter<uint32_t>("gttNumberOfPVs");
      channelSpecsInput_["gtt"] = l1t::demo::ChannelSpec{tmuxFactor_, gttTimeslices_, gttLatency_};
    }
    inputFileWriter_ =
        std::make_unique<l1t::demo::BoardDataWriter>(l1t::demo::parseFileFormat(fileFormat_),
                                                     iConfig.getParameter<std::string>("inputFileName"),
                                                     nInputFramesPerBX_,
                                                     tmuxFactor_,
                                                     iConfig.getParameter<uint32_t>("maxLinesPerInputFile"),
                                                     channelIdsInput_,
                                                     channelSpecsInput_);
  }

  if (writeOutputs_) {
    nOutputFramesPerBX_ = iConfig.getParameter<uint32_t>("nOutputFramesPerBX");

    outputRegions_ = iConfig.getParameter<std::vector<uint32_t>>("outputRegions");
    outputLinksPuppi_ = iConfig.getParameter<std::vector<uint32_t>>("outputLinksPuppi");
    for (unsigned int i = 0; i < outputLinksPuppi_.size(); ++i) {
      channelIdsOutput_[l1t::demo::LinkId{"puppi", i}].push_back(outputLinksPuppi_[i]);
    }
    channelSpecsOutput_["puppi"] = {tmuxFactor_, 0};
    nPuppiFramesPerRegion_ = (nOutputFramesPerBX_ * tmuxFactor_) / outputRegions_.size();
    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal) {
      outputBoard_ = iConfig.getParameter<int32_t>("outputBoard");
      outputLinkEgamma_ = iConfig.getParameter<int32_t>("outputLinkEgamma");
      nEgammaObjectsOut_ = iConfig.getParameter<uint32_t>("nEgammaObjectsOut");
      if (outputLinkEgamma_ != -1) {
        channelIdsOutput_[l1t::demo::LinkId{"egamma", 0}].push_back(outputLinkEgamma_);
        channelSpecsOutput_["egamma"] = {tmuxFactor_, nOutputFramesPerBX_ * tmuxFactor_ - 3 * nEgammaObjectsOut_};
      }
    }
    if ((outputBoard_ == -1) != (outputLinkEgamma_ == -1)) {
      throw cms::Exception("Configuration", "Inconsistent configuration of outputLinkEgamma, outputBoard");
    }
    outputFileWriter_ =
        std::make_unique<l1t::demo::BoardDataWriter>(l1t::demo::parseFileFormat(fileFormat_),
                                                     iConfig.getParameter<std::string>("outputFileName"),
                                                     nOutputFramesPerBX_,
                                                     tmuxFactor_,
                                                     iConfig.getParameter<uint32_t>("maxLinesPerOutputFile"),
                                                     channelIdsOutput_,
                                                     channelSpecsOutput_);
  }
}

L1TCorrelatorLayer1PatternFileWriter::~L1TCorrelatorLayer1PatternFileWriter() {}

void L1TCorrelatorLayer1PatternFileWriter::write(const l1ct::Event& event) {
  if (writeInputs_) {
    l1t::demo::EventData inputs;
    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal) {
      writeTF(event, inputs);
    }
    if (partition_ == Partition::Barrel) {
      writeBarrelGCT(event, inputs);
    }
    if (partition_ == Partition::HGCal || partition_ == Partition::HGCalNoTk) {
      writeHGC(event, inputs);
    }
    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal || partition_ == Partition::HGCalNoTk) {
      writeGMT(event, inputs);
    }
    if (partition_ == Partition::Barrel || partition_ == Partition::HGCal) {
      writeGTT(event, inputs);
    }
    inputFileWriter_->addEvent(inputs);
  }

  if (writeOutputs_) {
    l1t::demo::EventData outputs;
    writePuppi(event, outputs);
    if (outputLinkEgamma_ != -1)
      writeEgamma(event, outputs);
    outputFileWriter_->addEvent(outputs);
  }

  eventIndex_++;
  if (eventIndex_ % eventsPerFile_ == 0) {
    if (writeInputs_)
      inputFileWriter_->flush();
    if (writeOutputs_)
      outputFileWriter_->flush();
  }
}

L1TCorrelatorLayer1PatternFileWriter::Partition L1TCorrelatorLayer1PatternFileWriter::parsePartition(
    const std::string& partition) {
  if (partition == "Barrel")
    return Partition::Barrel;
  if (partition == "HGCal")
    return Partition::HGCal;
  if (partition == "HGCalNoTk")
    return Partition::HGCalNoTk;
  if (partition == "HF")
    return Partition::HF;
  throw cms::Exception("Configuration", "Unsupported partition_ '" + partition + "'\n");
}

void L1TCorrelatorLayer1PatternFileWriter::configTimeSlices(const edm::ParameterSet& iConfig,
                                                            const std::string& prefix,
                                                            unsigned int nSectors,
                                                            unsigned int nTimeSlices,
                                                            unsigned int linksFactor) {
  if (nTimeSlices > 1) {
    auto timeSliceConfig = iConfig.getParameter<std::vector<edm::ParameterSet>>(prefix + "TimeSlices");
    if (timeSliceConfig.size() != nTimeSlices)
      throw cms::Exception("Configuration")
          << "Mismatched number of " << prefix << "TimeSlices, expected " << nTimeSlices << std::endl;
    for (unsigned int iT = 0; iT < nTimeSlices; ++iT) {
      configSectors(timeSliceConfig[iT], prefix, nSectors, linksFactor);
    }
  } else {
    configSectors(iConfig, prefix, nSectors, linksFactor);
  }
}

void L1TCorrelatorLayer1PatternFileWriter::configSectors(const edm::ParameterSet& iConfig,
                                                         const std::string& prefix,
                                                         unsigned int nSectors,
                                                         unsigned int linksFactor) {
  if (nSectors > 1) {
    auto sectorConfig = iConfig.getParameter<std::vector<edm::ParameterSet>>(prefix + "Sectors");
    if (sectorConfig.size() != nSectors)
      throw cms::Exception("Configuration")
          << "Mismatched number of " << prefix << "Sectors, expected " << nSectors << std::endl;
    for (unsigned int iS = 0; iS < nSectors; ++iS) {
      configLinks(sectorConfig[iS], prefix, linksFactor, linksFactor > 1 ? iS * 10 : iS);
    }
  } else {
    configLinks(iConfig, prefix, linksFactor, 0);
  }
}
void L1TCorrelatorLayer1PatternFileWriter::configLinks(const edm::ParameterSet& iConfig,
                                                       const std::string& prefix,
                                                       unsigned int linksFactor,
                                                       unsigned int offset) {
  if (linksFactor > 1) {
    auto links = iConfig.getParameter<std::vector<int32_t>>(prefix + "Links");
    if (links.size() != linksFactor)
      throw cms::Exception("Configuration")
          << "Mismatched number of " << prefix << "Links, expected " << linksFactor << std::endl;
    for (unsigned int i = 0; i < linksFactor; ++i) {
      if (links[i] != -1) {
        channelIdsInput_[l1t::demo::LinkId{prefix, i + offset}].push_back(links[i]);
      }
    }
  } else {
    auto link = iConfig.getParameter<int32_t>(prefix + "Link");
    if (link != -1) {
      channelIdsInput_[l1t::demo::LinkId{prefix, offset}].push_back(link);
    }
  }
}

void L1TCorrelatorLayer1PatternFileWriter::writeTF(const l1ct::Event& event, l1t::demo::EventData& out) {
  for (unsigned int iS = 0, nS = event.raw.track.size(); iS < nS; ++iS) {
    l1t::demo::LinkId key{"tf", iS};
    if (channelIdsInput_.count(key) == 0)
      continue;
    std::vector<ap_uint<64>> ret;
    std::vector<ap_uint<96>> tracks = event.raw.track[iS].obj;
    if (tracks.empty())
      tracks.emplace_back(0);
    for (unsigned int i = 0, n = tracks.size(); i < n; ++i) {
      const ap_uint<96>& packedtk = tracks[i];
      if (i % 2 == 0) {
        ret.emplace_back(packedtk(63, 0));
        ret.emplace_back(0);
        ret.back()(31, 0) = packedtk(95, 64);
      } else {
        ret.back()(63, 32) = packedtk(31, 0);
        ret.emplace_back(packedtk(95, 32));
      }
    }
    out.add(key, ret);
  }
}

void L1TCorrelatorLayer1PatternFileWriter::writeHGC(const l1ct::Event& event, l1t::demo::EventData& out) {
  assert(hgcLinksFactor_ == 4);  // this piece of code won't really work otherwise
  std::vector<ap_uint<64>> ret[hgcLinksFactor_];
  for (unsigned int iS = 0, nS = event.raw.hgcalcluster.size(); iS < nS; ++iS) {
    l1t::demo::LinkId key0{"hgc", iS * 10};
    if (channelIdsInput_.count(key0) == 0)
      continue;
    for (unsigned int il = 0; il < hgcLinksFactor_; ++il) {
      // put header word and (dummy) towers
      ret[il].resize(31);
      ap_uint<64>& head64 = ret[il][0];
      head64(63, 48) = 0xABC0;                 // Magic
      head64(47, 38) = 0;                      // Opaque
      head64(39, 32) = (eventIndex_ % 3) * 6;  // TM slice
      head64(31, 24) = iS;                     // Sector
      head64(23, 16) = il;                     // link
      head64(15, 0) = eventIndex_ % 3564;      // BX
      for (unsigned int j = 0; j < 30; ++j) {
        ret[il][j + 1] = 4 * j + il;
      }
    }
    for (auto clust : event.raw.hgcalcluster[iS].obj) {
      for (unsigned int il = 0; il < hgcLinksFactor_; ++il) {
        ret[il].push_back(clust(64 * il + 63, 64 * il));
      }
    }
    for (unsigned int il = 0; il < hgcLinksFactor_; ++il) {
      out.add(l1t::demo::LinkId{"hgc", iS * 10 + il}, ret[il]);
    }
  }
}

void L1TCorrelatorLayer1PatternFileWriter::writeBarrelGCT(const l1ct::Event& event, l1t::demo::EventData& out) {
  std::vector<ap_uint<64>> ret;
  for (unsigned int iS = 0; iS < gctSectors_; ++iS) {
    l1t::demo::LinkId key0{"gct", iS * 10};
    if (channelIdsInput_.count(key0) == 0)
      continue;
    const auto& had = event.decoded.hadcalo[iS];
    const auto& ecal = event.decoded.emcalo[iS];
    unsigned int iLink = 0, nHad = had.size(), nEcal = ecal.size();
    for (unsigned int i = 0; i < gctLinksHad_; ++i, ++iLink) {
      ret.clear();
      for (unsigned int iHad = i; i < nHad; iHad += gctLinksHad_) {
        ret.emplace_back(had[iHad].pack());
      }
      if (ret.empty())
        ret.emplace_back(0);
      out.add(l1t::demo::LinkId{"gct", iS * 10 + iLink}, ret);
    }
    for (unsigned int i = 0; i < gctLinksEcal_; ++i, ++iLink) {
      ret.clear();
      for (unsigned int iEcal = i; i < nEcal; iEcal += gctLinksEcal_) {
        ret.emplace_back(ecal[iEcal].pack());
      }
      if (ret.empty())
        ret.emplace_back(0);
      out.add(l1t::demo::LinkId{"gct", iS * 10 + iLink}, ret);
    }
  }
}

void L1TCorrelatorLayer1PatternFileWriter::writeGMT(const l1ct::Event& event, l1t::demo::EventData& out) {
  l1t::demo::LinkId key{"gmt", 0};
  if (channelIdsInput_.count(key) == 0)
    return;
  std::vector<ap_uint<64>> muons = event.raw.muon.obj;
  muons.resize(gmtNumberOfMuons_, ap_uint<64>(0));
  out.add(key, muons);
}

void L1TCorrelatorLayer1PatternFileWriter::writeGTT(const l1ct::Event& event, l1t::demo::EventData& out) {
  l1t::demo::LinkId key{"gtt", 0};
  if (channelIdsInput_.count(key) == 0)
    return;
  std::vector<ap_uint<64>> pvs = event.pvs_emu;
  pvs.resize(gttNumberOfPVs_, ap_uint<64>(0));
  out.add(key, pvs);
}

void L1TCorrelatorLayer1PatternFileWriter::writePuppi(const l1ct::Event& event, l1t::demo::EventData& out) {
  unsigned int n = outputLinksPuppi_.size();
  std::vector<std::vector<ap_uint<64>>> links(n);
  for (auto ir : outputRegions_) {
    auto puppi = event.out[ir].puppi;
    unsigned int npuppi = puppi.size();
    for (unsigned int i = 0; i < n * nPuppiFramesPerRegion_; ++i) {
      links[i / nPuppiFramesPerRegion_].push_back(i < npuppi ? puppi[i].pack() : ap_uint<l1ct::PuppiObj::BITWIDTH>(0));
    }
  }
  for (unsigned int i = 0; i < n; ++i) {
    out.add(l1t::demo::LinkId{"puppi", i}, links[i]);
  }
}

void L1TCorrelatorLayer1PatternFileWriter::writeEgamma(const l1ct::Event& event, l1t::demo::EventData& out) {
  std::vector<ap_uint<64>> ret;
  const auto& pho = event.board_out[outputBoard_].egphoton;
  const auto& ele = event.board_out[outputBoard_].egelectron;
  ret.reserve(3 * nEgammaObjectsOut_);
  for (const auto& p : pho) {
    ret.emplace_back(p.pack());
  }
  ret.resize(nEgammaObjectsOut_, ap_uint<64>(0));
  for (const auto& p : ele) {
    ap_uint<128> dword = p.pack();
    ret.push_back(dword(63, 0));
    ret.push_back(dword(127, 64));
  }
  ret.resize(3 * nEgammaObjectsOut_, ap_uint<64>(0));
  out.add(l1t::demo::LinkId{"egamma", 0}, ret);
}

void L1TCorrelatorLayer1PatternFileWriter::flush() {
  if (inputFileWriter_)
    inputFileWriter_->flush();
  if (outputFileWriter_)
    outputFileWriter_->flush();
}

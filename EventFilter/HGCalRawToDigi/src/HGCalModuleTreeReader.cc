/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Pedro Silva, CERN
 *   Laurent Forthomme, CERN
 *
 ****************************************************************************/

#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalModuleTreeReader.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataDefinitions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TChain.h"

using namespace hgcal::econd;

HGCalModuleTreeReader::HGCalModuleTreeReader(const EmulatorParameters& params,
                                             const std::string& tree_name,
                                             const std::vector<std::string>& filenames)
    : Emulator(params) {
  TChain chain(tree_name.data());
  for (const auto& filename : filenames)
    chain.Add(filename.c_str());

  HGCModuleTreeEvent event;
  chain.SetBranchAddress("event", &event.event);
  chain.SetBranchAddress("chip", &event.chip);
  chain.SetBranchAddress("half", &event.half);
  chain.SetBranchAddress("daqdata", &event.daqdata);
  chain.SetBranchAddress("bxcounter", &event.bxcounter);
  chain.SetBranchAddress("eventcounter", &event.eventcounter);
  chain.SetBranchAddress("orbitcounter", &event.orbitcounter);
  chain.SetBranchAddress("trigtime", &event.trigtime);
  chain.SetBranchAddress("trigwidth", &event.trigwidth);

  for (long long i = 0; i < chain.GetEntries(); ++i) {
    chain.GetEntry(i);

    // check if event already exists
    EventId key{(uint32_t)event.eventcounter, (uint32_t)event.bxcounter, (uint32_t)event.orbitcounter};
    if (data_.count(key) == 0)
      data_[key] = ERxInput{};

    // check if chip already exists
    ERxId_t erxKey{(uint8_t)event.chip, (uint8_t)event.half};
    if (data_[key].count(erxKey) == 0)
      data_[key][erxKey] = ERxData{};

    // daqdata: header, CM, 37 ch, CRC32, idle
    if (const auto nwords = event.daqdata->size(); nwords != 41)
      throw cms::Exception("HGCalModuleTreeReader")
          << "Invalid number of words retrieved for event {" << event.eventcounter << ":" << event.bxcounter << ":"
          << event.orbitcounter << "}: should be 41, got " << nwords << ".";

    // 1st word is the header: discard
    // 2nd word are the common mode words

    const uint32_t cmword(event.daqdata->at(1));
    if (((cmword >> 20) & 0xfff) != 0)
      throw cms::Exception("HGCalModuleTreeReader")
          << "Consistency check failed for common mode word: " << ((cmword >> 20) & 0xfff) << " != 0.";

    data_[key][erxKey].cm1 = cmword & 0x3ff;
    data_[key][erxKey].cm0 = (cmword >> 10) & 0x3ff;

    // next 37 words are channel data
    for (size_t i = 2; i < 2 + params_.num_channels_per_erx; i++) {
      HGCROCChannelDataFrame<uint32_t> frame(0, event.daqdata->at(i));
      const auto tctp = static_cast<ToTStatus>(frame.tctp());
      data_[key][erxKey].tctp.push_back(tctp);
      data_[key][erxKey].adcm.push_back(frame.adcm1());
      data_[key][erxKey].adc.push_back(tctp == ToTStatus::ZeroSuppressed ? frame.adc() : 0);
      data_[key][erxKey].tot.push_back(tctp == ToTStatus::ZeroSuppressed ? frame.rawtot() : 0);
      data_[key][erxKey].toa.push_back(frame.toa());
    }

    // copy CRC32
    data_[key][erxKey].crc32 = event.daqdata->at(39);

    // we could assert the idle word from #40 if needed

    // copy metadata
    data_[key][erxKey].meta.push_back(event.trigtime);
    data_[key][erxKey].meta.push_back(event.trigwidth);
  }

  edm::LogInfo("HGCalModuleTreeReader") << "read " << data_.size() << " events.";

  it_data_ = data_.begin();
}

//
ECONDInput HGCalModuleTreeReader::next() {
  if (it_data_ == data_.end())
    throw cms::Exception("HGCalModuleTreeReader") << "Insufficient number of events were retrieved from input tree to "
                                                     "proceed with the generation of emulated events.";

  ++it_data_;
  return ECONDInput{it_data_->first, it_data_->second};
}

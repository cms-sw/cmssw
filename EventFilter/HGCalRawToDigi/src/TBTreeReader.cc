#include "EventFilter/HGCalRawToDigi/interface/TBTreeReader.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace hgcal::econd {
  TBTreeReader::TBTreeReader(const EmulatorParameters& params,
                             const std::string& tree_name,
                             const std::vector<std::string>& filenames)
      : Emulator(params), chain_(tree_name.c_str()) {
    for (const auto& filename : filenames)
      chain_.Add(filename.c_str());
    TreeEvent event;
    chain_.SetBranchAddress("event", &event.event);
    chain_.SetBranchAddress("chip", &event.chip);
    chain_.SetBranchAddress("half", &event.half);
    chain_.SetBranchAddress("channel", &event.channel);
    chain_.SetBranchAddress("adc", &event.adc);
    chain_.SetBranchAddress("adcm", &event.adcm);
    chain_.SetBranchAddress("toa", &event.toa);
    chain_.SetBranchAddress("tot", &event.tot);
    chain_.SetBranchAddress("totflag", &event.totflag);
    chain_.SetBranchAddress("bxcounter", &event.bxcounter);
    chain_.SetBranchAddress("eventcounter", &event.eventcounter);
    chain_.SetBranchAddress("orbitcounter", &event.orbitcounter);

    for (long long i = 0; i < chain_.GetEntries(); i++) {
      chain_.GetEntry(i);
      // check if event already exists
      EventId key((uint32_t)event.eventcounter, (uint32_t)event.bxcounter, (uint32_t)event.orbitcounter);
      if (data_.count(key) == 0)
        data_[key] = ERxEvent();
      // check if chip already exists
      ERx_t erxKey((uint8_t)event.chip, (uint8_t)event.half);
      if (data_[key].count(erxKey) == 0)
        data_[key][erxKey] = ERxData();
      // add channel data
      if (event.channel == (int)params_.num_channels_per_erx)
        data_[key][erxKey].cm0 = event.adc;
      else if (event.channel == (int)params_.num_channels_per_erx + 1)
        data_[key][erxKey].cm1 = event.adc;
      else {
        data_[key][erxKey].tctp.push_back(event.totflag ? 3 : 0);
        data_[key][erxKey].adc.push_back(event.adc);
        data_[key][erxKey].tot.push_back(event.tot);
        data_[key][erxKey].adcm.push_back(event.adcm);
        data_[key][erxKey].toa.push_back(event.toa);
      }
    }
    it_data_ = data_.begin();
  }

  ECONDEvent TBTreeReader::next() {
    if (it_data_ == data_.end())
      throw cms::Exception("TBTreeReader") << "Insufficient number of events were retrieved from input tree to "
                                              "proceed with the generation of emulated events.";
    ++it_data_;
    return std::make_pair(it_data_->first, it_data_->second);
  }
}  // namespace hgcal::econd

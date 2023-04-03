#include "EventFilter/HGCalRawToDigi/interface/HGCalModuleTreeReader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HGCalDigi/interface/HGCROCChannelDataFrame.h"

using namespace hgcal::econd;

HGCalModuleTreeReader::HGCalModuleTreeReader(const EmulatorParameters& params,
                                             const std::string& tree_name,
                                             const std::vector<std::string>& filenames)
    : Emulator(params), chain_(tree_name.c_str()) {
  for (const auto& filename : filenames)
    chain_.Add(filename.c_str());

  HGCModuleTreeEvent event;
  chain_.SetBranchAddress("event", &event.event);
  chain_.SetBranchAddress("chip", &event.chip);
  chain_.SetBranchAddress("half", &event.half);
  chain_.SetBranchAddress("daqdata", &event.daqdata);
  chain_.SetBranchAddress("bxcounter", &event.bxcounter);
  chain_.SetBranchAddress("eventcounter", &event.eventcounter);
  chain_.SetBranchAddress("orbitcounter", &event.orbitcounter);
  chain_.SetBranchAddress("trigtime", &event.trigtime);
  chain_.SetBranchAddress("trigwidth", &event.trigwidth);

  for (long long i = 0; i < chain_.GetEntries(); i++) {
    chain_.GetEntry(i);

    // check if event already exists
    EventId key((uint32_t)event.eventcounter, (uint32_t)event.bxcounter, (uint32_t)event.orbitcounter);
    if (data_.count(key) == 0)
      data_[key] = ERxInput();

    // check if chip already exists
    ERxId_t erxKey((uint8_t)event.chip, (uint8_t)event.half);
    if (data_[key].count(erxKey) == 0)
      data_[key][erxKey] = ERxData();

    //daqdata: header, CM, 37 ch, CRC32, idle
    uint32_t nwords(event.daqdata->size());
    assert(nwords == 41);

    //1st word is the header: discard
    //2nd word are the common mode words
    uint32_t cmword(event.daqdata->at(1));
    data_[key][erxKey].cm1 = cmword & 0x3ff;
    data_[key][erxKey].cm0 = (cmword >> 10) & 0x3ff;
    assert(((cmword >> 20) & 0xfff) == 0);

    //next 37 words are channel data
    for (size_t i = 2; i < 39; i++) {
      HGCROCChannelDataFrame<uint32_t> frame(0, event.daqdata->at(i));
      uint16_t tctp(frame.tctp());
      data_[key][erxKey].tctp.push_back(tctp);
      data_[key][erxKey].adcm.push_back(frame.adcm1());
      data_[key][erxKey].adc.push_back(tctp == 0 ? frame.adc() : 0);
      data_[key][erxKey].tot.push_back(tctp == 0 ? frame.rawtot() : 0);
      data_[key][erxKey].toa.push_back(frame.toa());
    }

    //copy CRC32
    data_[key][erxKey].crc32 = event.daqdata->at(39);

    //we could assert the idle word from #40 if needed

    //copy metadata
    data_[key][erxKey].meta.push_back(event.trigtime);
    data_[key][erxKey].meta.push_back(event.trigwidth);
  }

  std::cout << "[HGCalModuleTreeReader] read " << data_.size() << " events" << std::endl;

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

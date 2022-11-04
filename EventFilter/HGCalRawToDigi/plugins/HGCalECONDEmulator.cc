#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "EventFilter/HGCalRawToDigi/interface/RawDataPackingTools.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulatorInfo.h"

#include "CLHEP/Random/RandFlat.h"
#include "TChain.h"

class HGCalECONDEmulator : public edm::stream::EDProducer<> {
public:
  explicit HGCalECONDEmulator(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  /// 8bit CRC for event header
  uint8_t computeCRC(const std::vector<uint32_t>&) const;

  struct HeaderBits_t {
    bool bitO, bitB, bitE, bitT, bitH, bitS;
  };
  std::vector<uint32_t> generateERxData(const edm::Event&, std::vector<uint64_t>&) const;
  std::vector<uint32_t> produceECONEvent(const edm::Event&, std::vector<uint64_t>&, HeaderBits_t&) const;

  const std::vector<std::string> input_files_;
  const std::vector<unsigned int> enabled_channels_;
  const unsigned int num_channels_;
  const unsigned int header_marker_;
  const unsigned int idle_marker_;
  const unsigned int fed_id_;

  const struct EmulatorParameters {
    explicit EmulatorParameters(const edm::ParameterSet&);
    double chan_surv_prob;
    double bitO_error_prob, bitB_error_prob, bitE_error_prob, bitT_error_prob, bitH_error_prob, bitS_error_prob;
  } params_;
  const bool store_emul_info_;

  std::unique_ptr<TChain> chain_;

  // input tree collections
  struct TreeEvent {
    int event, chip, half, channel, adc, adcm, toa, tot, totflag, bxcounter, eventcounter, orbitcounter;
  };
  typedef std::tuple<uint32_t, uint32_t, uint32_t> Event_t;
  typedef std::tuple<uint8_t, uint8_t> ERx_t;
  struct ERxData_t {
    std::vector<uint16_t> adc, adcm, toa, tot;
    std::vector<uint8_t> tctp;
    uint32_t cm0, cm1;
  };
  typedef std::map<Event_t, std::map<ERx_t, ERxData_t>> ECONDInputs_t;
  ECONDInputs_t data_;
  ECONDInputs_t::const_iterator it_data_;

  edm::Service<edm::RandomNumberGenerator> rng_;
  edm::EDPutTokenT<FEDRawDataCollection> fedRawToken_;
  edm::EDPutTokenT<HGCalECONDEmulatorInfo> fedEmulInfoToken_;
};

HGCalECONDEmulator::HGCalECONDEmulator(const edm::ParameterSet& iConfig)
    : input_files_(iConfig.getParameter<std::vector<std::string>>("inputs")),
      enabled_channels_(iConfig.getParameter<std::vector<unsigned int>>("enabledChannels")),
      num_channels_(iConfig.getParameter<unsigned int>("numChannels")),
      header_marker_(iConfig.getParameter<unsigned int>("headerMarker")),
      idle_marker_(iConfig.getParameter<unsigned int>("idleMarker")),
      fed_id_(iConfig.getParameter<unsigned int>("fedId")),
      params_(iConfig.getParameter<edm::ParameterSet>("probabilityMaps")),
      store_emul_info_(iConfig.getParameter<bool>("storeEmulatorInfo")),
      chain_(new TChain(iConfig.getParameter<std::string>("treeName").data())) {
  if (!rng_.isAvailable())
    throw cms::Exception("HGCalECONDEmulator") << "The HGCalECONDEmulator module requires the "
                                                  "RandomNumberGeneratorService,\n"
                                                  "which appears to be absent. Please add that service to your "
                                                  "configuration\n"
                                                  "or remove the modules that require it.";

  fedRawToken_ = produces<FEDRawDataCollection>();
  if (store_emul_info_)
    fedEmulInfoToken_ = produces<HGCalECONDEmulatorInfo>();
}

void HGCalECONDEmulator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  if (it_data_ == data_.end())
    throw cms::Exception("HGCalECONDEmulator") << "Insufficient number of events were retrieved from input tree to "
                                                  "proceed with the generation of emulated events.";

  std::vector<uint64_t> enabled_channels;
  HeaderBits_t header_bits;

  auto gen_event = produceECONEvent(iEvent, enabled_channels, header_bits);

  // convert 32-bit event into 64-bit payloads
  std::vector<uint64_t> packed_event;
  for (size_t i = 0; i < gen_event.size(); i += 2)
    packed_event.emplace_back((uint64_t(gen_event.at(i)) << 32) | gen_event.at(i + 1));
  size_t event_size = packed_event.size() * sizeof(packed_event.at(0)) * sizeof(unsigned char);

  // fill the output FED raw data collection
  FEDRawDataCollection raw_data;
  auto& fed_data = raw_data.FEDData(fed_id_);
  fed_data.resize(event_size + FEDHeader::length + FEDTrailer::length);
  auto* ptr = fed_data.data();

  int trg_type = 0;  //FIXME
  const auto event_id = std::get<0>(it_data_->first), bx_id = std::get<1>(it_data_->first);

  // compose 2*32-bit FED header word
  FEDHeader::set(ptr, trg_type, event_id, bx_id, fed_id_);
  LogDebug("HGCalECONDEmulator").log([&](auto& log) {
    const FEDHeader hdr(ptr);
    log << "FED header: lvl1ID=" << hdr.lvl1ID() << ", bxID=" << hdr.bxID() << ", source ID=" << hdr.sourceID() << ".";
  });
  ptr += FEDHeader::length;

  // insert ECON-D payload
  LogDebug("HGCalECONDEmulator") << "Will write " << packed_event.size() << " 64-bit words = " << event_size
                                 << " 8-bit words.";
  std::memcpy(ptr, packed_event.data(), event_size);
  ptr += event_size;

  // compose 2*32-bit FED trailer word
  FEDTrailer::set(
      ptr, packed_event.size() + 2, evf::compute_crc(reinterpret_cast<uint8_t*>(packed_event.data()), event_size), 0, 0);
  LogDebug("HGCalECONDEmulator").log([&](auto& log) {
    const FEDTrailer trl(ptr);
    log << "FED trailer: fragment length: " << trl.fragmentLength() << ", CRC=0x" << std::hex << trl.crc() << std::dec
        << ", status: " << trl.evtStatus() << ".";
  });
  ptr += FEDTrailer::length;

  iEvent.emplace(fedRawToken_, std::move(raw_data));

  // store the emulation information if requested
  if (store_emul_info_) {
    HGCalECONDEmulatorInfo emul_info(header_bits.bitO,
                                     header_bits.bitB,
                                     header_bits.bitE,
                                     header_bits.bitT,
                                     header_bits.bitH,
                                     header_bits.bitS,
                                     enabled_channels);
    iEvent.emplace(fedEmulInfoToken_, std::move(emul_info));
  }
  ++it_data_;
}

std::vector<uint32_t> HGCalECONDEmulator::produceECONEvent(const edm::Event& iEvent,
                                                           std::vector<uint64_t>& enabled_channels,
                                                           HeaderBits_t& header_bits) const {
  auto* rng_engine = &rng_->getEngine(iEvent.streamID());

  // first sample on header status bits
  header_bits.bitO = CLHEP::RandFlat::shoot(rng_engine) >= params_.bitO_error_prob;
  header_bits.bitB = CLHEP::RandFlat::shoot(rng_engine) >= params_.bitB_error_prob;
  header_bits.bitE = CLHEP::RandFlat::shoot(rng_engine) >= params_.bitE_error_prob;
  header_bits.bitT = CLHEP::RandFlat::shoot(rng_engine) >= params_.bitT_error_prob;
  header_bits.bitH = CLHEP::RandFlat::shoot(rng_engine) >= params_.bitH_error_prob;
  header_bits.bitS = CLHEP::RandFlat::shoot(rng_engine) >= params_.bitS_error_prob;

  auto econ_event = generateERxData(iEvent, enabled_channels);
  LogDebug("HGCalECONDEmulator") << econ_event.size() << " word(s) of eRx payloads inserted.";

  // as ECON-D event content was just created, only prepend packet header at
  // this stage
  auto econdH = hgcal::econd::eventPacketHeader(
      header_marker_,
      econ_event.size() + 1,
      true,
      false,
      // HGCROC Event reco status across all active eRxE-B-O:
      // FIXME check endianness of these two numbers
      (header_bits.bitH << 1) | header_bits.bitT,                            // HDR/TRL numbers
      (header_bits.bitE << 2) | (header_bits.bitB << 1) | header_bits.bitO,  // Event/BX/Orbit numbers
      false,
      false,
      0,
      std::get<0>(it_data_->first),
      std::get<1>(it_data_->first),
      std::get<2>(it_data_->first),
      header_bits.bitS,  // OR of "Stat" bits for all active eRx
      0,
      0);
  econ_event.insert(econ_event.begin(), econdH.begin(), econdH.end());
  LogDebug("HGCalECONDEmulator") << econdH.size() << " word(s) of event packet header prepend. New size of ECON frame: "
                                 << econ_event.size();

  econ_event.push_back(computeCRC(econdH));
  econ_event.push_back(hgcal::econd::buildIdleWord(0, 0, 0, idle_marker_));

  return econ_event;
}

std::vector<uint32_t> HGCalECONDEmulator::generateERxData(const edm::Event& iEvent,
                                                          std::vector<uint64_t>& enabled_channels) const {
  auto* rng_engine = &rng_->getEngine(iEvent.streamID());

  enabled_channels.clear();  // reset the list of channels enabled

  std::vector<uint32_t> erxData;
  for (const auto& jt : it_data_->second) {
    std::vector<bool> chmap(num_channels_, false);
    uint64_t ch_en = 0ull;  // reset the list of channels enabled
    for (size_t i = 0; i < chmap.size(); i++) {
      // randomly choosing the channels to be shot at
      chmap[i] = (enabled_channels_.empty() ||
                  (std::find(enabled_channels_.begin(), enabled_channels_.end(), i) != enabled_channels_.end())) &&
                 CLHEP::RandFlat::shoot(rng_engine) <= params_.chan_surv_prob;
      ch_en += (chmap[i] << i);
    }
    enabled_channels.emplace_back(ch_en & ((1 << (num_channels_ + 1)) - 1));  // mask only (num_channels_) LSBs

    auto erxHeader = hgcal::econd::eRxSubPacketHeader(0, 0, false, jt.second.cm0, jt.second.cm1, chmap);
    erxData.insert(erxData.end(), erxHeader.begin(), erxHeader.end());
    for (size_t i = 0; i < num_channels_; i++) {
      if (!chmap.at(i))
        continue;
      uint8_t msb = 32;
      auto chData = hgcal::econd::addChannelData(msb,
                                                 jt.second.tctp.at(i),
                                                 jt.second.adc.at(i),
                                                 jt.second.tot.at(i),
                                                 jt.second.adcm.at(i),
                                                 jt.second.toa.at(i),
                                                 true,
                                                 true,
                                                 true,
                                                 true);
      erxData.insert(erxData.end(), chData.begin(), chData.end());
    }
  }
  return erxData;
}

uint8_t HGCalECONDEmulator::computeCRC(const std::vector<uint32_t>& event_header) const {
  uint8_t crc = 0;  //FIXME 8-bit Bluetooth CRC
  return crc;
}

void HGCalECONDEmulator::beginStream(edm::StreamID) {
  for (const auto& input : input_files_)
    chain_->Add(input.data());

  TreeEvent event;
  chain_->SetBranchAddress("event", &event.event);
  chain_->SetBranchAddress("chip", &event.chip);
  chain_->SetBranchAddress("half", &event.half);
  chain_->SetBranchAddress("channel", &event.channel);
  chain_->SetBranchAddress("adc", &event.adc);
  chain_->SetBranchAddress("adcm", &event.adcm);
  chain_->SetBranchAddress("toa", &event.toa);
  chain_->SetBranchAddress("tot", &event.tot);
  chain_->SetBranchAddress("totflag", &event.totflag);
  chain_->SetBranchAddress("bxcounter", &event.bxcounter);
  chain_->SetBranchAddress("eventcounter", &event.eventcounter);
  chain_->SetBranchAddress("orbitcounter", &event.orbitcounter);

  for (long long i = 0; i < chain_->GetEntries(); i++) {
    chain_->GetEntry(i);
    // check if event already exists
    Event_t key((uint32_t)event.eventcounter, (uint32_t)event.bxcounter, (uint32_t)event.orbitcounter);
    if (data_.count(key) == 0)
      data_[key] = std::map<ERx_t, ERxData_t>();
    // check if chip already exists
    ERx_t erxKey((uint8_t)event.chip, (uint8_t)event.half);
    if (data_[key].count(erxKey) == 0)
      data_[key][erxKey] = ERxData_t();
    // add channel data
    if (event.channel == (int)num_channels_)
      data_[key][erxKey].cm0 = event.adc;
    else if (event.channel == (int)num_channels_ + 1)
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

HGCalECONDEmulator::EmulatorParameters::EmulatorParameters(const edm::ParameterSet& iConfig)
    : chan_surv_prob(iConfig.getParameter<double>("channelSurv")),
      bitO_error_prob(iConfig.getParameter<double>("bitOError")),
      bitB_error_prob(iConfig.getParameter<double>("bitBError")),
      bitE_error_prob(iConfig.getParameter<double>("bitEError")),
      bitT_error_prob(iConfig.getParameter<double>("bitTError")),
      bitH_error_prob(iConfig.getParameter<double>("bitHError")),
      bitS_error_prob(iConfig.getParameter<double>("bitSError")) {}

void HGCalECONDEmulator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("treeName", "unpacker_data/hgcroc");
  desc.add<std::vector<std::string>>("inputs", {})
      ->setComment("list of input files containing HGCROC emulated/test beam frames");
  desc.add<std::vector<unsigned int>>("enabledChannels", {})->setComment("list of channels to be enabled in readout");
  desc.add<unsigned int>("numChannels", 37)->setComment("number of channels managed in ECON-D");
  desc.add<unsigned int>("headerMarker", 0x154)->setComment("9b programmable pattern; default is '0xAA' + '0b0'");
  desc.add<unsigned int>("idleMarker", 0x555500);
  desc.add<unsigned int>("fedId", 0)->setComment("FED number delivering the emulated frames");

  edm::ParameterSetDescription prob_desc;
  prob_desc.add<double>("channelSurv", 1.);
  prob_desc.add<double>("bitOError", 0.);
  prob_desc.add<double>("bitBError", 0.);
  prob_desc.add<double>("bitEError", 0.);
  prob_desc.add<double>("bitTError", 0.);
  prob_desc.add<double>("bitHError", 0.);
  prob_desc.add<double>("bitSError", 0.);
  desc.add<edm::ParameterSetDescription>("probabilityMaps", prob_desc);

  desc.add<bool>("storeEmulatorInfo", true)
      ->setComment("also append a 'truth' auxiliary info to the output event content");
  descriptions.add("hgcalEmulatedECONDRawData", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalECONDEmulator);

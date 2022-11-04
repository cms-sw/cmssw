#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "EventFilter/HGCalRawToDigi/interface/HGCalFrameGenerator.h"
#include "EventFilter/HGCalRawToDigi/interface/RawDataPackingTools.h"

#include "CLHEP/Random/RandFlat.h"

namespace hgcal {
  HGCalFrameGenerator::HGCalFrameGenerator(const edm::ParameterSet& iConfig)
      : pass_through_(iConfig.getParameter<bool>("passThroughMode")),
        expected_mode_(iConfig.getParameter<bool>("expectedMode")),
        matching_ebo_numbers_(iConfig.getParameter<bool>("matchingEBOnumbers")),
        bo_truncated_(iConfig.getParameter<bool>("bufferOverflowTruncated")) {
    const auto econd_params = iConfig.getParameter<edm::ParameterSet>("econdParams");
    econd_.chan_surv_prob = econd_params.getParameter<double>("channelSurv");
    econd_.enabled_channels = econd_params.getParameter<std::vector<unsigned int> >("enabledChannels");
    econd_.header_marker = econd_params.getParameter<unsigned int>("headerMarker");
    econd_.num_channels = econd_params.getParameter<unsigned int>("numChannels");
    econd_.bitO_error_prob = econd_params.getParameter<double>("bitOError");
    econd_.bitB_error_prob = econd_params.getParameter<double>("bitBError");
    econd_.bitE_error_prob = econd_params.getParameter<double>("bitEError");
    econd_.bitT_error_prob = econd_params.getParameter<double>("bitTError");
    econd_.bitH_error_prob = econd_params.getParameter<double>("bitHError");
    econd_.bitS_error_prob = econd_params.getParameter<double>("bitSError");

    const auto slink_params = iConfig.getParameter<edm::ParameterSet>("slinkParams");
    slink_.num_econds = slink_params.getParameter<unsigned int>("numECONDs");
  }

  edm::ParameterSetDescription HGCalFrameGenerator::description() {
    edm::ParameterSetDescription desc;

    desc.add<bool>("passThroughMode", true)->setComment("ECON-D in pass-through mode?");
    desc.add<bool>("expectedMode", false)->setComment("is an Event HDR/TRL expected to be received from the HGCROCs?");
    desc.add<bool>("matchingEBOnumbers", false)
        ->setComment(
            "is the transmitted E/B/O (according to mode selected by user) matching the E/B/O value in the ECON-D L1A "
            "FIFO?");
    desc.add<bool>("bufferOverflowTruncated", false)->setComment("is the packet truncated for buffer overflow?");

    edm::ParameterSetDescription econd_desc;
    econd_desc.add<double>("channelSurv", 1.);
    econd_desc.add<std::vector<unsigned int> >("enabledChannels", {})
        ->setComment("list of channels to be enabled in readout");
    econd_desc.add<unsigned int>("headerMarker", 0x154)
        ->setComment("9b programmable pattern; default is '0xAA' + '0b0'");
    econd_desc.add<unsigned int>("numChannels", 37)->setComment("number of channels managed in ECON-D");
    econd_desc.add<double>("bitOError", 0.);
    econd_desc.add<double>("bitBError", 0.);
    econd_desc.add<double>("bitEError", 0.);
    econd_desc.add<double>("bitTError", 0.);
    econd_desc.add<double>("bitHError", 0.);
    econd_desc.add<double>("bitSError", 0.);
    desc.add<edm::ParameterSetDescription>("econdParams", econd_desc);

    edm::ParameterSetDescription slink_desc;
    slink_desc.add<unsigned int>("numECONDs", 7);
    desc.add<edm::ParameterSetDescription>("slinkParams", slink_desc);

    return desc;
  }

  void HGCalFrameGenerator::setRandomEngine(CLHEP::HepRandomEngine& rng) { rng_ = &rng; }

  HGCalFrameGenerator::HeaderBits HGCalFrameGenerator::generateStatusBits() const {
    HeaderBits header_bits;
    // first sample on header status bits
    header_bits.bitO = CLHEP::RandFlat::shoot(rng_) >= econd_.bitO_error_prob;
    header_bits.bitB = CLHEP::RandFlat::shoot(rng_) >= econd_.bitB_error_prob;
    header_bits.bitE = CLHEP::RandFlat::shoot(rng_) >= econd_.bitE_error_prob;
    header_bits.bitT = CLHEP::RandFlat::shoot(rng_) >= econd_.bitT_error_prob;
    header_bits.bitH = CLHEP::RandFlat::shoot(rng_) >= econd_.bitH_error_prob;
    header_bits.bitS = CLHEP::RandFlat::shoot(rng_) >= econd_.bitS_error_prob;
    return header_bits;
  }

  std::vector<bool> HGCalFrameGenerator::generateEnabledChannels(uint64_t& ch_en) const {
    std::vector<bool> chmap(econd_.num_channels, false);
    ch_en = 0ull;  // reset the list of channels enabled
    for (size_t i = 0; i < chmap.size(); i++) {
      // randomly choosing the channels to be shot at
      chmap[i] = (econd_.enabled_channels.empty() ||
                  (std::find(econd_.enabled_channels.begin(), econd_.enabled_channels.end(), i) !=
                   econd_.enabled_channels.end())) &&
                 CLHEP::RandFlat::shoot(rng_) <= econd_.chan_surv_prob;
      ch_en += (chmap[i] << i);
    }
    ch_en &= ((1 << (econd_.num_channels + 1)) - 1);  // mask only (econd_.num_channels) LSBs
    return chmap;
  }

  std::vector<uint32_t> HGCalFrameGenerator::generateERxData(const econd::ERxEvent& event,
                                                             std::vector<uint64_t>& enabled_channels) const {
    enabled_channels.clear();  // reset the list of channels enabled

    std::vector<uint32_t> erxData;
    for (const auto& jt : event) {
      uint64_t ch_en;  // list of channels enabled
      auto chmap = generateEnabledChannels(ch_en);
      enabled_channels.emplace_back(ch_en);

      auto erxHeader = econd::eRxSubPacketHeader(0, 0, false, jt.second.cm0, jt.second.cm1, chmap);
      erxData.insert(erxData.end(), erxHeader.begin(), erxHeader.end());
      for (size_t i = 0; i < econd_.num_channels; i++) {
        if (!chmap.at(i))
          continue;
        uint8_t msb = 32;
        auto chData = econd::addChannelData(msb,
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

  std::vector<uint32_t> HGCalFrameGenerator::produceECONEvent(const econd::ECONDEvent& event) const {
    std::vector<uint64_t> enabled_channels;
    auto header_bits = generateStatusBits();
    auto econd_event = generateERxData(event.second, enabled_channels);
    LogDebug("HGCalFrameGenerator") << econd_event.size() << " word(s) of eRx payloads inserted.";

    last_econd_emul_info_.clear();
    // as ECON-D event content was just created, only prepend packet header at
    // this stage
    auto econdH = hgcal::econd::eventPacketHeader(
        econd_.header_marker,
        econd_event.size() + 1,
        pass_through_,
        expected_mode_,
        // HGCROC Event reco status across all active eRxE-B-O:
        // FIXME check endianness of these two numbers
        (header_bits.bitH << 1) | header_bits.bitT,                            // HDR/TRL numbers
        (header_bits.bitE << 2) | (header_bits.bitB << 1) | header_bits.bitO,  // Event/BX/Orbit numbers
        matching_ebo_numbers_,
        bo_truncated_,
        0,
        std::get<0>(event.first),
        std::get<1>(event.first),
        std::get<2>(event.first),
        header_bits.bitS,  // OR of "Stat" bits for all active eRx
        0,
        0);
    econd_event.insert(econd_event.begin(), econdH.begin(), econdH.end());
    LogDebug("HGCalFrameGenerator") << econdH.size()
                                    << " word(s) of event packet header prepend. New size of ECON frame: "
                                    << econd_event.size();

    econd_event.push_back(computeCRC(econdH));

    return econd_event;
  }

  std::vector<uint64_t> HGCalFrameGenerator::produceSlinkEvent(const econd::ECONDEvent& econd_event) const {
    std::vector<uint64_t> slink_event;

    const auto& event_id = econd_event.first;
    uint64_t oc = std::get<2>(event_id), ec = std::get<0>(event_id), bc = std::get<1>(event_id);

    uint64_t l1a_header{0ul};
    l1a_header |= ((oc & 0xf) << 36);
    l1a_header |= ((ec & 0x3f) << 40);
    l1a_header |= ((bc & 0xfff) << 46);

    last_slink_emul_info_.clear();
    for (size_t i = 0; i < max_num_econds_; ++i) {
      if (i < slink_.num_econds) {
        auto econd_evt = produceECONEvent(econd_event);
        const auto& econd_evt_info = lastECONDEmulatedInfo();
        for (size_t j = 0; j < econd_evt.size(); j += 2) {
          uint64_t word1 = econd_evt.at(j), word2 = (j + 1 < econd_evt.size()) ? econd_evt.at(j + 1) : 0ul;
          slink_event.emplace_back(((word1 & 0xffffffff) << 32) | (word2 & 0xffffffff));
        }
        uint8_t econd_packet_status = ECONDPacketStatus::Normal;  //FIXME
        l1a_header |= ((econd_packet_status & 0x7) << (3 * i));
        last_slink_emul_info_.addECONDEmulatedInfo(econd_evt_info);
      } else {
        l1a_header |= (ECONDPacketStatus::InactiveECOND << (3 * i));
      }
    }
    slink_event.insert(slink_event.begin(), l1a_header);  // prepend L1A header

    return slink_event;
  }

  uint8_t HGCalFrameGenerator::computeCRC(const std::vector<uint32_t>& event_header) const {
    uint8_t crc = 0;  //FIXME 8-bit Bluetooth CRC
    return crc;
  }
}  // namespace hgcal

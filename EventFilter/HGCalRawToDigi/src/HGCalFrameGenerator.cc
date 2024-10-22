/****************************************************************************
 *
 * This is a part of HGCAL offline software.
 * Authors:
 *   Laurent Forthomme, CERN
 *
 ****************************************************************************/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulator.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalFrameGenerator.h"
#include "EventFilter/HGCalRawToDigi/interface/HGCalRawDataPackingTools.h"

#include "CLHEP/Random/RandFlat.h"
#include <iomanip>

namespace hgcal {
  //--------------------------------------------------
  // bit-casting utilities
  //--------------------------------------------------

  template <typename T>
  void printWords(edm::MessageSender& os, const std::string& name, const std::vector<T> vec) {
    os << "Dump of the '" << name << "' words:\n";
    for (size_t i = 0; i < vec.size(); ++i)
      os << std::dec << std::setfill(' ') << std::setw(4) << i << ": 0x" << std::hex << std::setfill('0')
         << std::setw(sizeof(T) * 2) << vec.at(i) << "\n";
  }

  static std::vector<uint64_t> to64bit(const std::vector<uint32_t>& in) {
    std::vector<uint64_t> out;
    for (size_t i = 0; i < in.size(); i += 2) {
      uint64_t word1 = (i < in.size()) ? in.at(i) : 0ul, word2 = (i + 1 < in.size()) ? in.at(i + 1) : 0ul;
      out.emplace_back(((word2 & 0xffffffff) << 32) | (word1 & 0xffffffff));
    }
    return out;
  }

  static std::vector<uint64_t> to128bit(const std::vector<uint64_t>& in) {
    std::vector<uint64_t> out;
    for (size_t i = 0; i < in.size(); i += 2) {
      out.emplace_back(in.at(i));
      out.emplace_back((i + 1 < in.size()) ? in.at(i + 1) : 0u);
    }
    return out;
  }

  HGCalFrameGenerator::HGCalFrameGenerator(const edm::ParameterSet& iConfig) {
    const auto slink_config = iConfig.getParameter<edm::ParameterSet>("slinkParams");

    size_t econd_id = 0;
    std::vector<unsigned int> active_econds;
    for (const auto& econd : slink_config.getParameter<std::vector<edm::ParameterSet> >("ECONDs")) {
      // a bit of user input validation
      if (slink_config.getParameter<bool>("checkECONDsLimits")) {
        if (active_econds.size() > kMaxNumECONDs)
          throw cms::Exception("HGCalFrameGenerator")
              << "Too many active ECON-D set: " << active_econds.size() << " > " << kMaxNumECONDs << ".";
        if (econd_id >= kMaxNumECONDs)
          throw cms::Exception("HGCalFrameGenerator")
              << "Invalid ECON-D identifier: " << econd_id << " >= " << kMaxNumECONDs << ".";
      }
      if (econd.getParameter<bool>("active"))
        active_econds.emplace_back(econd_id);

      econd_params_.insert(std::make_pair(econd_id, econd::EmulatorParameters(econd)));
      ++econd_id;
    }

    slink_params_ = SlinkParameters{.active_econds = active_econds,
                                    .boe_marker = slink_config.getParameter<unsigned int>("boeMarker"),
                                    .eoe_marker = slink_config.getParameter<unsigned int>("eoeMarker"),
                                    .format_version = slink_config.getParameter<unsigned int>("formatVersion"),
                                    .num_capture_blocks = slink_config.getParameter<unsigned int>("numCaptureBlocks"),
                                    .store_header_trailer = slink_config.getParameter<bool>("storeHeaderTrailer")};
  }

  edm::ParameterSetDescription HGCalFrameGenerator::description() {
    edm::ParameterSetDescription desc;

    std::vector<edm::ParameterSet> econds_psets;
    for (size_t i = 0; i < 7; ++i)
      econds_psets.emplace_back();

    edm::ParameterSetDescription slink_desc;
    slink_desc.addVPSet("ECONDs", econd::EmulatorParameters::description(), econds_psets)
        ->setComment("list of active ECON-Ds in S-link");
    slink_desc.add<unsigned int>("boeMarker", 0x55);
    slink_desc.add<unsigned int>("eoeMarker", 0xaa);
    slink_desc.add<unsigned int>("formatVersion", 3);
    slink_desc.add<unsigned int>("numCaptureBlocks", 1)
        ->setComment("number of capture blocks to emulate per S-link payload");
    slink_desc.add<bool>("checkECONDsLimits", true)->setComment("check the maximal number of ECON-Ds per S-link");
    slink_desc.add<bool>("storeHeaderTrailer", true)->setComment("also store the S-link header and trailer words");
    desc.add<edm::ParameterSetDescription>("slinkParams", slink_desc);

    return desc;
  }

  void HGCalFrameGenerator::setRandomEngine(CLHEP::HepRandomEngine& rng) { rng_ = &rng; }

  void HGCalFrameGenerator::setEmulator(econd::Emulator& emul) { emul_ = &emul; }

  //--------------------------------------------------
  // emulation part
  //--------------------------------------------------

  HGCalFrameGenerator::HeaderBits HGCalFrameGenerator::generateStatusBits(unsigned int econd_id) const {
    if (!rng_)
      throw cms::Exception("HGCalFrameGenerator") << "Random number generator not initialised.";
    const auto& econd_params = econd_params_.at(econd_id);
    // first sample on header status bits
    return HeaderBits{
        .bitO = CLHEP::RandFlat::shoot(rng_) < econd_params.error_prob.bitO,
        .bitB = CLHEP::RandFlat::shoot(rng_) < econd_params.error_prob.bitB,
        .bitE = CLHEP::RandFlat::shoot(rng_) < econd_params.error_prob.bitE,
        .bitT = CLHEP::RandFlat::shoot(rng_) < econd_params.error_prob.bitT,
        .bitH = CLHEP::RandFlat::shoot(rng_) < econd_params.error_prob.bitH,
        .bitS = CLHEP::RandFlat::shoot(rng_) < econd_params.error_prob.bitS,
    };
  }

  econd::ERxChannelEnable HGCalFrameGenerator::generateEnabledChannels(unsigned int econd_id) const {
    const auto& econd_params = econd_params_.at(econd_id);
    econd::ERxChannelEnable chmap(econd_params.num_channels_per_erx, false);
    for (size_t i = 0; i < chmap.size(); i++)
      // randomly choosing the channels to be shot at
      chmap[i] = CLHEP::RandFlat::shoot(rng_) <= econd_params.chan_surv_prob;
    return chmap;
  }

  std::vector<uint32_t> HGCalFrameGenerator::generateERxData(
      unsigned int econd_id,
      const econd::ERxInput& input_event,
      std::vector<econd::ERxChannelEnable>& enabled_channels) const {
    const auto& econd_params = econd_params_.at(econd_id);
    std::vector<uint32_t> erx_data;
    enabled_channels.clear();
    for (const auto& jt : input_event) {  // one per eRx
      const auto chmap =
          generateEnabledChannels(econd_id);  // generate a list of probable channels to be filled with emulated content

      // insert eRx header (common mode, channels map, ...)
      uint8_t stat = 0b111 /*possibly emulate*/, hamming_check = 0;
      bool bit_e = false;  // did unmasked stat error bits associated with the eRx cause the sub-packet to be supressed?
      auto erx_header = econd::eRxSubPacketHeader(stat, hamming_check, bit_e, jt.second.cm0, jt.second.cm1, chmap);
      erx_data.insert(erx_data.end(), erx_header.begin(), erx_header.end());
      if (jt.second.adc.size() < econd_params.num_channels_per_erx) {
        edm::LogError("HGCalFrameGenerator:generateERxData")
            << "Data multiplicity too low (" << jt.second.adc.size() << ") to emulate "
            << econd_params.num_channels_per_erx << " ECON-D channel(s).";
        continue;
      }
      // insert eRx payloads (integrating all readout channels)
      const auto erx_chan_data = econd::produceERxData(chmap,
                                                       jt.second,
                                                       true,  // passZS
                                                       true,  // passZSm1
                                                       true,  // hasToA
                                                       econd_params.characterisation_mode);
      erx_data.insert(erx_data.end(), erx_chan_data.begin(), erx_chan_data.end());
      enabled_channels.emplace_back(chmap);
    }
    LogDebug("HGCalFrameGenerator").log([&erx_data](auto& log) { printWords(log, "erx", erx_data); });
    return erx_data;
  }

  uint32_t HGCalFrameGenerator::computeCRC(const std::vector<uint32_t>& event_header) const {
    uint32_t crc = 0x12345678;  //TODO: implement 32-bit Bluetooth CRC in the future
    return crc;
  }

  //--------------------------------------------------
  // payload creation utilities
  //--------------------------------------------------

  std::vector<uint64_t> HGCalFrameGenerator::produceECONEvent(unsigned int econd_id, unsigned int cb_id) const {
    if (!emul_)
      throw cms::Exception("HGCalFrameGenerator")
          << "ECON-D emulator was not properly set to the frame generator. Please ensure you are calling the "
             "HGCalFrameGenerator::setEmulator method.";

    std::vector<uint64_t> econd_event;

    const auto event = emul_->next();
    const auto& econd_params = econd_params_.at(econd_id);
    auto header_bits = generateStatusBits(econd_id);
    std::vector<econd::ERxChannelEnable> enabled_ch_per_erx;
    auto erx_payload = generateERxData(econd_id, event.second, enabled_ch_per_erx);

    // ECON-D event content was just created, now prepend packet header
    const uint8_t hamming = 0, rr = 0;
    auto econd_header =
        econd::eventPacketHeader(econd_params.header_marker,
                                 erx_payload.size() + 1 /*CRC*/,
                                 econd_params.passthrough_mode,
                                 econd_params.expected_mode,
                                 // HGCROC Event reco status across all active eRx E-B-O:
                                 (header_bits.bitH & 0x1) << 1 | (header_bits.bitT & 0x1),  // HDR/TRL numbers
                                 (header_bits.bitE & 0x1) << 2 | (header_bits.bitB & 0x1) << 1 |
                                     (header_bits.bitO & 0x1),  // Event/BX/Orbit numbers
                                 econd_params.matching_ebo_numbers,
                                 econd_params.bo_truncated,
                                 hamming,                   // Hamming for event header
                                 std::get<1>(event.first),  // BX
                                 std::get<0>(event.first),  // event id (L1A)
                                 std::get<2>(event.first),  // orbit
                                 header_bits.bitS,          // OR of "Stat" bits for all active eRx
                                 rr);
    LogDebug("HGCalFrameGenerator").log([&econd_header](auto& log) { printWords(log, "econ-d header", econd_header); });
    auto econd_header_64bit = to64bit(econd_header);
    econd_event.insert(econd_event.end(), econd_header_64bit.begin(), econd_header_64bit.end());
    LogDebug("HGCalFrameGenerator") << econd_header.size()
                                    << " word(s) of event packet header prepend. New size of ECON frame: "
                                    << econd_event.size();
    const auto erx_payload_64bit = to64bit(erx_payload);
    econd_event.insert(econd_event.end(), erx_payload_64bit.begin(), erx_payload_64bit.end());
    LogDebug("HGCalFrameGenerator") << erx_payload.size() << " word(s) of eRx payloads inserted.";

    std::vector<uint64_t> econd_footer;
    if (econd_params.add_econd_crc)
      econd_footer.emplace_back(computeCRC(econd_header));
    if (econd_params.add_idle_word) {
      const uint8_t buffer_status = 0, error_status = 0, reset_request = 0;
      econd_footer.emplace_back(
          econd::buildIdleWord(buffer_status, error_status, reset_request, econd_params.programmable_pattern));
    }
    econd_event.insert(econd_event.end(), econd_footer.begin(), econd_footer.end());
    // bookkeeping of last event + metadata
    last_slink_emul_info_.captureBlockEmulatedInfo(cb_id).addECONDEmulatedInfo(
        econd_id,
        HGCalECONDEmulatorInfo(header_bits.bitO,
                               header_bits.bitT,
                               header_bits.bitE,
                               header_bits.bitT,
                               header_bits.bitH,
                               header_bits.bitS,
                               enabled_ch_per_erx));
    last_emul_event_ = event;
    return econd_event;
  }

  std::vector<uint64_t> HGCalFrameGenerator::produceCaptureBlockEvent(unsigned int cb_id) const {
    std::vector<uint64_t> cb_event;
    // build all ECON-Ds payloads and add them to the capture block payload
    std::vector<backend::ECONDPacketStatus> econd_statuses(kMaxNumECONDs, backend::ECONDPacketStatus::InactiveECOND);
    for (const auto& econd : econd_params_) {  // for each ECON-D payload to be emulated
      const auto& econd_id = econd.first;
      if (!econd.second.active)
        continue;  // status is already inactive
      econd_statuses[econd_id] =
          backend::ECONDPacketStatus::Normal;  //TODO: also implement/emulate other ECON-D packet issues
      const auto econd_payload = produceECONEvent(econd_id, cb_id);
      cb_event.insert(cb_event.end(), econd_payload.begin(), econd_payload.end());
    }
    const auto& eid = last_emul_event_.first;
    const uint64_t event_id = std::get<0>(eid), bx_id = std::get<1>(eid), orbit_id = std::get<2>(eid);
    // prepend the header to the capture block payload
    const auto l1a_header = to64bit(backend::buildCaptureBlockHeader(bx_id, event_id, orbit_id, econd_statuses));
    LogDebug("HGCalFrameGenerator").log([&l1a_header](auto& log) { printWords(log, "l1a", l1a_header); });
    cb_event.insert(cb_event.begin(), l1a_header.begin(), l1a_header.end());
    return to128bit(cb_event);
  }

  std::vector<uint64_t> HGCalFrameGenerator::produceSlinkEvent(unsigned int fed_id) const {
    last_slink_emul_info_.clear();  // clear the metadata of the latest emulated S-link payload

    std::vector<uint64_t> slink_event;  // prepare the output S-link payload (will be "converted" to 128-bit)
    for (unsigned int cb_id = 0; cb_id < slink_params_.num_capture_blocks; ++cb_id) {
      const auto cb_payload = produceCaptureBlockEvent(cb_id);
      slink_event.insert(slink_event.end(),
                         cb_payload.begin(),
                         cb_payload.end());  // concatenate the capture block to the full S-link payload
    }

    if (slink_params_.store_header_trailer) {
      // build the S-link header words
      const uint32_t content_id = backend::buildSlinkContentId(backend::SlinkEmulationFlag::Subsystem, 0, 0);
      const auto& eid = last_emul_event_.first;
      const uint64_t event_id = std::get<0>(eid), bx_id = std::get<1>(eid), orbit_id = std::get<2>(eid);
      const auto slink_header = to128bit(to64bit(backend::buildSlinkHeader(
          slink_params_.boe_marker, slink_params_.format_version, event_id, content_id, fed_id)));
      slink_event.insert(slink_event.begin(), slink_header.begin(), slink_header.end());  // prepend S-link header

      // build the S-link trailer words
      const bool fed_crc_err = false, slinkrocket_crc_err = false, source_id_err = false, sync_lost = false,
                 fragment_trunc = false;
      const uint16_t status =
          backend::buildSlinkRocketStatus(fed_crc_err, slinkrocket_crc_err, source_id_err, sync_lost, fragment_trunc);

      const uint16_t daq_crc = 0, crc = 0;
      const uint32_t event_length = slink_event.size() - slink_header.size() - 1;
      const auto slink_trailer = to128bit(to64bit(
          backend::buildSlinkTrailer(slink_params_.eoe_marker, daq_crc, event_length, bx_id, orbit_id, crc, status)));
      slink_event.insert(slink_event.end(), slink_trailer.begin(), slink_trailer.end());  // append S-link trailer

      LogDebug("HGCalFrameGenerator").log([&slink_header, &slink_trailer](auto& log) {
        printWords(log, "slink header", slink_header);
        log << "\n";
        printWords(log, "slink trailer", slink_trailer);
      });
    }

    return to128bit(slink_event);
  }
}  // namespace hgcal

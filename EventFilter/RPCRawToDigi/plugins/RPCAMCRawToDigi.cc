#include "EventFilter/RPCRawToDigi/plugins/RPCAMCRawToDigi.h"

#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMCLinkEvents.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMCRecord.h"
#include "EventFilter/RPCRawToDigi/plugins/RPCAMCUnpacker.h"
#include "EventFilter/RPCRawToDigi/plugins/RPCAMCUnpackerFactory.h"

RPCAMCRawToDigi::RPCAMCRawToDigi(edm::ParameterSet const &config)
    : calculate_crc_(config.getParameter<bool>("calculateCRC")),
      fill_counters_(config.getParameter<bool>("fillCounters")),
      rpc_unpacker_(
          RPCAMCUnpackerFactory::get()->create(config.getParameter<std::string>("RPCAMCUnpacker"),
                                               config.getParameter<edm::ParameterSet>("RPCAMCUnpackerSettings"),
                                               producesCollector())) {
  if (fill_counters_) {
    produces<RPCAMCLinkCounters>();
  }

  raw_token_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("inputTag"));
}

RPCAMCRawToDigi::~RPCAMCRawToDigi() {}

void RPCAMCRawToDigi::fillDescriptions(edm::ConfigurationDescriptions &descs) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("rawDataCollector", ""));
  desc.add<bool>("calculateCRC", true);
  desc.add<bool>("fillCounters", true);
  desc.add<std::string>("RPCAMCUnpacker", "RPCAMCUnpacker");
  RPCAMCUnpacker::fillDescription(desc);
  descs.add("RPCAMCRawToDigi", desc);
}

void RPCAMCRawToDigi::beginRun(edm::Run const &run, edm::EventSetup const &setup) {
  rpc_unpacker_->beginRun(run, setup);
}

void RPCAMCRawToDigi::produce(edm::Event &event, edm::EventSetup const &setup) {
  // Get RAW Data
  edm::Handle<FEDRawDataCollection> raw_data_collection;
  event.getByToken(raw_token_, raw_data_collection);

  std::unique_ptr<RPCAMCLinkCounters> counters(new RPCAMCLinkCounters());

  std::map<RPCAMCLink, rpcamc13::AMCPayload> amc_payload;

  // Loop over the FEDs
  for (int fed : rpc_unpacker_->getFeds()) {
    if (fill_counters_) {
      counters->add(RPCAMCLinkEvents::fed_event_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
    }

    std::uint16_t crc(0xffff);

    FEDRawData const &raw_data = raw_data_collection->FEDData(fed);
    unsigned int nwords(raw_data.size() / sizeof(std::uint64_t));
    if (!nwords) {
      continue;
    }

    std::uint64_t const *word(reinterpret_cast<std::uint64_t const *>(raw_data.data()));
    std::uint64_t const *word_end = word + nwords;

    LogDebug("RPCAMCRawToDigi") << "Handling FED " << fed << " with length " << nwords;

    // Handle the CDF Headers
    if (!processCDFHeaders(fed, word, word_end, crc, *counters)) {
      continue;
    }

    // Handle the CDF Trailers
    if (!processCDFTrailers(fed, nwords, word, word_end, crc, *counters)) {
      continue;
    }

    // Loop over the Blocks
    if (!processBlocks(fed, word, word_end, crc, *counters, amc_payload)) {
      continue;
    }

    // Complete CRC check with trailer
    if (calculate_crc_) {
      word = word_end;
      word_end = reinterpret_cast<std::uint64_t const *>(raw_data.data()) + nwords - 1;
      for (; word < word_end; ++word) {
        compute_crc16_64bit(crc, *word);
      }
      compute_crc16_64bit(crc, (*word & 0xffffffff0000ffff));  // trailer excluding crc
      FEDTrailer trailer(reinterpret_cast<unsigned char const *>(word_end));
      if ((unsigned int)(trailer.crc()) != crc) {
        if (fill_counters_) {
          counters->add(RPCAMCLinkEvents::fed_trailer_crc_mismatch_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
        }
        edm::LogWarning("RPCAMCRawToDigi") << "FED Trailer CRC doesn't match for FED id " << fed;
        continue;
      }
    }
  }

  rpc_unpacker_->produce(event, setup, amc_payload);

  if (fill_counters_) {
    event.put(std::move(counters));
  }
}

bool RPCAMCRawToDigi::processCDFHeaders(int fed,
                                        std::uint64_t const *&word,
                                        std::uint64_t const *&word_end,
                                        std::uint16_t &crc,
                                        RPCAMCLinkCounters &counters) const {
  bool more_headers(true);
  for (; word < word_end && more_headers; ++word) {
    if (calculate_crc_) {
      compute_crc16_64bit(crc, *word);
    }

    LogDebug("RPCAMCRawToDigi") << "CDF Header " << std::hex << *word << std::dec;
    FEDHeader header(reinterpret_cast<unsigned char const *>(word));
    if (!header.check()) {
      if (fill_counters_) {
        counters.add(RPCAMCLinkEvents::fed_header_check_fail_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
      }
      edm::LogWarning("RPCAMCRawToDigi") << "FED Header check failed for FED id " << fed;
      ++word;
      return false;
    }
    if (header.sourceID() != fed) {
      if (fill_counters_) {
        counters.add(RPCAMCLinkEvents::fed_header_id_mismatch_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
      }
      edm::LogWarning("RPCAMCRawToDigi") << "FED Header Source ID " << header.sourceID()
                                         << " does not match requested FED id " << fed;
      ++word;
      return false;
    }

    // moreHeaders() not used
    // more_headers = header.moreHeaders();
    more_headers = false;
  }

  return !more_headers;
}

bool RPCAMCRawToDigi::processCDFTrailers(int fed,
                                         unsigned int nwords,
                                         std::uint64_t const *&word,
                                         std::uint64_t const *&word_end,
                                         std::uint16_t &crc,
                                         RPCAMCLinkCounters &counters) const {
  bool more_trailers(true);
  for (--word_end; word_end > word && more_trailers; --word_end) {
    FEDTrailer trailer(reinterpret_cast<unsigned char const *>(word_end));
    LogDebug("RPCAMCRawToDigi") << "CDF Trailer " << std::hex << *word_end << std::dec << ", length "
                                << trailer.fragmentLength();
    if (!trailer.check()) {
      if (fill_counters_) {
        counters.add(RPCAMCLinkEvents::fed_trailer_check_fail_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
      }
      edm::LogWarning("RPCAMCRawToDigi") << "FED Trailer check failed for FED id " << fed;
      return false;
    }
    if (trailer.fragmentLength() != nwords) {
      if (fill_counters_) {
        counters.add(RPCAMCLinkEvents::fed_trailer_length_mismatch_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
      }
      edm::LogWarning("RPCAMCRawToDigi") << "FED Trailer length " << trailer.fragmentLength()
                                         << " does not match actual data size " << nwords << " for FED id " << fed;
      return false;
    }

    // moreTrailers() not used
    // more_trailers = trailer.moreTrailers();
    more_trailers = false;
  }

  ++word_end;

  return !more_trailers;
}

bool RPCAMCRawToDigi::processBlocks(int fed,
                                    std::uint64_t const *&word,
                                    std::uint64_t const *word_end,
                                    std::uint16_t &crc,
                                    RPCAMCLinkCounters &counters,
                                    std::map<RPCAMCLink, rpcamc13::AMCPayload> &amc_payload) const {
  while (word < word_end) {
    rpcamc13::Header header(*word);
    if (calculate_crc_) {
      compute_crc16_64bit(crc, *word);
    }
    ++word;

    unsigned int n_amc(header.getNAMC());
    if (word + n_amc + 1 >= word_end) {  // AMC Headers and AMC13 Trailer
      if (fill_counters_) {
        counters.add(RPCAMCLinkEvents::fed_amc13_block_incomplete_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
      }
      edm::LogWarning("RPCAMCRawToDigi") << "AMC13 Block can not be complete for FED " << fed;
      word = word_end;
      return false;
    }

    std::uint64_t const *payload_word(word + n_amc);
    std::uint64_t const *payload_word_end(payload_word);
    for (unsigned int amc = 0; amc < n_amc; ++amc) {
      rpcamc13::AMCHeader amc13_amc_header(*word);
      if (calculate_crc_) {
        compute_crc16_64bit(crc, *word);
      }
      ++word;

      unsigned int size_in_block(amc13_amc_header.getSizeInBlock());
      if (size_in_block == 0) {
        continue;
      }
      payload_word = payload_word_end;
      payload_word_end += size_in_block;

      unsigned int amc_number(amc13_amc_header.getAMCNumber());
      if (amc_number > (unsigned int)RPCAMCLink::max_amcnumber_) {
        if (fill_counters_) {
          counters.add(RPCAMCLinkEvents::fed_amc13_amc_number_invalid_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
        }
        edm::LogWarning("RPCAMCRawToDigi") << "Invalid AMC Number " << amc_number << " for FED " << fed;
        continue;
      }

      if (payload_word_end > word_end) {
        if (fill_counters_) {
          counters.add(RPCAMCLinkEvents::amc_amc13_block_incomplete_, RPCAMCLink(fed, amc_number));
        }
        edm::LogWarning("RPCAMCRawToDigi")
            << "AMC Block can not be complete for FED " << fed << " at AMC " << amc_number;
        return false;
      }

      rpcamc13::AMCPayload &payload(amc_payload[RPCAMCLink(fed, amc_number)]);

      if (amc13_amc_header.isFirstBlock()) {
        payload.setAMCHeader(amc13_amc_header);
        if (fill_counters_) {
          counters.add(RPCAMCLinkEvents::amc_event_, RPCAMCLink(fed, amc_number));
        }

        if (!amc13_amc_header.isValid()) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_amc13_evc_bc_invalid_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi")
              << "AMC13 AMC Header is reporting an invalid "
              << "Event Counter or Bunch Counter for FED " << fed << ", AMC " << amc_number;
          payload.setValid(false);
        }

        rpcamc::Header amc_amc_header(payload_word);
        if (amc_number != amc_amc_header.getAMCNumber()) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_number_mismatch_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi")
              << "AMC Number inconsistent in AMC13 AMC Header vs AMC Header: " << amc_number << " vs "
              << amc_amc_header.getAMCNumber();
          payload.setValid(false);
        }

        if (amc_amc_header.hasDataLength() && amc13_amc_header.hasTotalSize() &&
            amc13_amc_header.getSize() != amc_amc_header.getDataLength()) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_size_mismatch_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi")
              << "AMC size inconsistent in AMC13 AMC Header vs AMC Header: " << amc13_amc_header.getSize() << " vs "
              << amc_amc_header.getDataLength();
          payload.setValid(false);
        }
      }

      if (amc13_amc_header.isLastBlock()) {
        if (!amc13_amc_header.isFirstBlock()) {
          edm::LogWarning("RPCAMCRawToDigi") << "Multiple blocks";
        }
        payload.getAMCHeader().setCRCOk(amc13_amc_header.isCRCOk());
        payload.getAMCHeader().setLengthCorrect(amc13_amc_header.isLengthCorrect());

        if (!amc13_amc_header.isCRCOk()) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_amc13_crc_mismatch_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi")
              << "AMC13 AMC Header is reporting a mismatched  "
              << "Event Counter or Bunch Counter for FED " << fed << ", AMC " << amc_number;
          payload.setValid(false);
        }

        if (!amc13_amc_header.isLengthCorrect()) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_amc13_length_incorrect_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi") << "AMC13 AMC Header is reporting an incorrect length "
                                             << "for FED " << fed << ", AMC " << amc_number;
          payload.setValid(false);
        }

        if (amc13_amc_header.hasTotalSize() &&
            amc13_amc_header.getSize() != (payload.getData().size() + size_in_block)) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_amc13_size_inconsistent_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi") << "Size in AMC13 AMC Header doesn't match payload size "
                                             << "for FED " << fed << ", AMC " << amc_number;
          payload.setValid(false);
        }

        if (!payload.getData().empty() && (payload.getData().size() + size_in_block) < 3) {
          if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_payload_incomplete_, RPCAMCLink(fed, amc_number));
          }
          edm::LogWarning("RPCAMCRawToDigi") << "Size in AMC13 AMC Header doesn't match payload size "
                                             << "for FED " << fed << ", AMC " << amc_number;
          payload.setValid(false);
        }
      }

      if (size_in_block > 0) {
        payload.insert(payload_word, size_in_block);
      }
    }

    if (calculate_crc_) {
      for (; word < payload_word_end; ++word) {
        compute_crc16_64bit(crc, *word);
      }
    } else {
      word = payload_word_end;
    }

    if (calculate_crc_) {
      compute_crc16_64bit(crc, *word);
    }
    ++word;
  }
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCAMCRawToDigi);

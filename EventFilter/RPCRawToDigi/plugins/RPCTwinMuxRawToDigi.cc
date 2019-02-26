#include "EventFilter/RPCRawToDigi/plugins/RPCTwinMuxRawToDigi.h"

#include <memory>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/CRC16.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMCLinkEvents.h"

RPCTwinMuxRawToDigi::RPCTwinMuxRawToDigi(edm::ParameterSet const & config)
    : calculate_crc_(config.getParameter<bool>("calculateCRC"))
    , fill_counters_(config.getParameter<bool>("fillCounters"))
    , bx_min_(config.getParameter<int>("bxMin"))
    , bx_max_(config.getParameter<int>("bxMax"))
{
    produces<RPCDigiCollection>();
    if (fill_counters_) {
        produces<RPCAMCLinkCounters>();
    }
    raw_token_ = consumes<FEDRawDataCollection>(config.getParameter<edm::InputTag>("inputTag"));
}

RPCTwinMuxRawToDigi::~RPCTwinMuxRawToDigi()
{}

void RPCTwinMuxRawToDigi::compute_crc_64bit(std::uint16_t & crc, std::uint64_t const & word)
{ // overcome constness problem evf::compute_crc_64bit
    unsigned char const * uchars(reinterpret_cast<unsigned char const *>(&word));
    for (unsigned char const * uchar = uchars + 7
             ; uchar >= uchars ; --uchar) {
        crc = evf::compute_crc_8bit(crc, *uchar);
    }
}

void RPCTwinMuxRawToDigi::fillDescriptions(edm::ConfigurationDescriptions & descs)
{
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputTag", edm::InputTag("rawDataCollector", ""));
    desc.add<bool>("calculateCRC", true);
    desc.add<bool>("fillCounters", true);
    desc.add<int>("bxMin", -2);
    desc.add<int>("bxMax", 2);
    descs.add("rpcTwinMuxRawToDigi", desc);
}

void RPCTwinMuxRawToDigi::beginRun(edm::Run const & run, edm::EventSetup const & setup)
{
    if (es_tm_link_map_watcher_.check(setup)) {
        setup.get<RPCTwinMuxLinkMapRcd>().get(es_tm_link_map_);
        std::set<int> feds;
        for (auto const & tm_link : es_tm_link_map_->getMap()) {
            feds.insert(tm_link.first.getFED());
        }
        feds_.assign(feds.begin(), feds.end());
    }
}

void RPCTwinMuxRawToDigi::produce(edm::Event & event, edm::EventSetup const & setup)
{
    // Get EventSetup Electronics Maps
    setup.get<RPCTwinMuxLinkMapRcd>().get(es_tm_link_map_);
    setup.get<RPCLBLinkMapRcd>().get(es_lb_link_map_);

    // Get RAW Data
    edm::Handle<FEDRawDataCollection> raw_data_collection;
    event.getByToken(raw_token_, raw_data_collection);

    std::set<std::pair<RPCDetId, RPCDigi> > rpc_digis;
    std::unique_ptr<RPCAMCLinkCounters> counters(new RPCAMCLinkCounters());

    // Loop over the FEDs
    for (int fed : feds_) {

        if (fill_counters_) {
            counters->add(RPCAMCLinkEvents::fed_event_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
        }

        std::uint16_t crc(0xffff);

        FEDRawData const & raw_data = raw_data_collection->FEDData(fed);
        unsigned int nwords(raw_data.size() / sizeof(std::uint64_t));
        if (!nwords) {
            continue;
        }

        std::uint64_t const * word(reinterpret_cast<std::uint64_t const *>(raw_data.data()));
        std::uint64_t const * word_end = word + nwords;

        LogDebug("RPCTwinMuxRawToDigi") << "Handling FED " << fed << " with length " << nwords;

        // Handle the CDF Headers
        if (!processCDFHeaders(fed
                               , word, word_end
                               , crc, *counters)) {
            continue;
        }

        // Handle the CDF Trailers
        if (!processCDFTrailers(fed, nwords
                                , word, word_end
                                , crc, *counters)) {
            continue;
        }

        // Loop over the Blocks
        while (word < word_end) {
            processBlock(fed
                         , word, word_end
                         , crc, *counters, rpc_digis);
        }

        // Complete CRC check with trailer
        if (calculate_crc_) {
            word = word_end;
            word_end = reinterpret_cast<std::uint64_t const *>(raw_data.data()) + nwords - 1;
            for ( ; word < word_end ; ++word) {
                compute_crc_64bit(crc, *word);
            }
            compute_crc_64bit(crc, (*word & 0xffffffff0000ffff)); // trailer excluding crc
            FEDTrailer trailer(reinterpret_cast<unsigned char const *>(word_end));
            if ((unsigned int)(trailer.crc()) != crc) {
                if (fill_counters_) {
                    counters->add(RPCAMCLinkEvents::fed_trailer_crc_mismatch_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
                }
                edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Trailer CRC doesn't match for FED id " << fed;
            }
        }
    }

    putRPCDigis(event, rpc_digis);
    if (fill_counters_) {
        putCounters(event, std::move(counters));
    }
}

bool RPCTwinMuxRawToDigi::processCDFHeaders(int fed
                                            , std::uint64_t const * & word, std::uint64_t const * & word_end
                                            , std::uint16_t & crc
                                            , RPCAMCLinkCounters & counters) const
{
    bool more_headers(true);
    for ( ; word < word_end && more_headers ; ++word) {
        if (calculate_crc_) {
            compute_crc_64bit(crc, *word);
        }

        LogDebug("RPCTwinMuxRawToDigi") << "CDF Header " << std::hex << *word << std::dec;
        FEDHeader header(reinterpret_cast<unsigned char const *>(word));
        if (!header.check()) {
            if (fill_counters_) {
                counters.add(RPCAMCLinkEvents::fed_header_check_fail_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Header check failed for FED id " << fed;
            ++word;
            break;
        }
        if (header.sourceID() != fed) {
            if (fill_counters_) {
                counters.add(RPCAMCLinkEvents::fed_header_id_mismatch_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Header Source ID " << header.sourceID()
                                                   << " does not match requested FED id " << fed;
            break;
        }

        // moreHeaders() not used
        // more_headers = header.moreHeaders();
        more_headers = false;
    }

    return !more_headers;
}

bool RPCTwinMuxRawToDigi::processCDFTrailers(int fed, unsigned int nwords
                                             , std::uint64_t const * & word, std::uint64_t const * & word_end
                                             , std::uint16_t & crc
                                             , RPCAMCLinkCounters & counters) const
{
    bool more_trailers(true);
    for (--word_end ; word_end > word && more_trailers ; --word_end) {
        FEDTrailer trailer(reinterpret_cast<unsigned char const *>(word_end));
        LogDebug("RPCTwinMuxRawToDigi") << "CDF Trailer " << std::hex << *word_end << std::dec
                                        << ", length " << trailer.fragmentLength();
        if (!trailer.check()) {
            if (fill_counters_) {
                counters.add(RPCAMCLinkEvents::fed_trailer_check_fail_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Trailer check failed for FED id " << fed;
            --word_end;
            break;
        }
        if (trailer.fragmentLength() != nwords) {
            if (fill_counters_) {
                counters.add(RPCAMCLinkEvents::fed_trailer_length_mismatch_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Trailer length " << trailer.fragmentLength()
                                                   << " does not match actual data size " << nwords
                                                   << " for FED id " << fed;
            --word_end;
            break;
        }

        // moreTrailers() not used
        // more_trailers = trailer.moreTrailers();
        more_trailers = false;
    }

    ++word_end;

    return !more_trailers;
}

bool RPCTwinMuxRawToDigi::processBlock(int fed
                                       , std::uint64_t const * & word, std::uint64_t const * word_end
                                       , std::uint16_t & crc
                                       , RPCAMCLinkCounters & counters
                                       , std::set<std::pair<RPCDetId, RPCDigi> > & digis) const
{
    // Block Header and Content
    rpctwinmux::BlockHeader block_header(*word);
    if (calculate_crc_) {
        compute_crc_64bit(crc, *word);
    }
    ++word;

    unsigned int n_amc(block_header.getNAMC());
    if (word + n_amc + 1 >= word_end) {
        if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::fed_amc13_block_incomplete_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "Block can not be complete for FED " << fed;
        word = word_end;
        return false;
    }

    std::vector<std::pair<unsigned int, unsigned int> > amc_size_map;
    for (unsigned int amc = 0 ; amc < n_amc ; ++amc) {
        LogDebug("RPCTwinMuxRawToDigi") << "Block AMC " << amc;
        rpctwinmux::BlockAMCContent amc_content(*word);
        if (calculate_crc_) {
            compute_crc_64bit(crc, *word);
        }
        ++word;

        amc_size_map.push_back(std::make_pair(amc_content.getAMCNumber(), amc_content.getSize()));
        if (!amc_content.isValid()) {
            if (fill_counters_) {
                counters.add(RPCAMCLinkEvents::amc_amc13_evc_bc_invalid_, RPCAMCLink(fed, amc_content.getAMCNumber()));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "BlockAMCContent is reporting an invalid "
                                                   << "Event Counter or Bunch Counter for FED " << fed
                                                   << ", AMC " << amc_content.getAMCNumber();
        }
    }

    for (std::pair<unsigned int, unsigned int> const & amc_size : amc_size_map) {
        processTwinMux(fed, amc_size.first, amc_size.second
                       , word, word_end
                       , crc, counters
                       , digis);
    }

    if (word < word_end) {
        rpctwinmux::BlockTrailer block_trailer(*word);
        if (calculate_crc_) {
            compute_crc_64bit(crc, *word);
        }
        ++word;
        return true;
    } else {
        return false;
    }
}

bool RPCTwinMuxRawToDigi::processTwinMux(int fed, unsigned int amc_number, unsigned int size
                                         , std::uint64_t const * & word, std::uint64_t const * word_end
                                         , std::uint16_t & crc
                                         , RPCAMCLinkCounters & counters
                                         , std::set<std::pair<RPCDetId, RPCDigi> > & digis) const
{
    LogDebug("RPCTwinMuxRawToDigi") << "TwinMux AMC#" << amc_number << ", size " << size;
    if (!size) {
        return true;
    }
    if (amc_number > (unsigned int)RPCAMCLink::max_amcnumber_) {
        if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::fed_amc13_amc_number_invalid_, RPCAMCLink(fed, RPCAMCLink::wildcard_));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "Invalid AMC Number " << amc_number
                                               << " for FED " << fed;
        if (calculate_crc_) {
            for ( ; size > 0 ; --size, ++word) {
                compute_crc_64bit(crc, *word);
            }
        } else {
            word += size;
        }
        return false;
    }
    if (word + size >= word_end || size < 3) {
        if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_payload_incomplete_, RPCAMCLink(fed, amc_number));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "TwinMux Data can not be complete for FED " << fed << " AMC #" << amc_number;
        if (calculate_crc_) {
            for ( ; size > 0 ; --size, ++word) {
                compute_crc_64bit(crc, *word);
            }
        } else {
            word += size;
        }
        return false;
    }

    rpctwinmux::TwinMuxHeader header(word);
    unsigned int bx_counter(header.getBXCounter());
    if (calculate_crc_) {
        compute_crc_64bit(crc, *word); ++word;
        compute_crc_64bit(crc, *word); ++word;
    } else {
        word += 2;
    }
    size -= 2;

    if (amc_number != header.getAMCNumber()) {
        if (fill_counters_) {
            counters.add(RPCAMCLinkEvents::amc_number_mismatch_, RPCAMCLink(fed, amc_number));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "AMC Number inconsistent in TwinMuxHeader vs BlockAMCContent: " << header.getAMCNumber()
                                               << " vs " << amc_number;
        if (calculate_crc_) {
            for ( ; size > 0 ; --size, ++word) {
                compute_crc_64bit(crc, *word);
            }
        } else {
            word += size;
        }
        return false;
    }

    int bx_min(bx_min_), bx_max(bx_max_);
    if (header.hasRPCBXWindow()) {
        bx_min = std::max(bx_min, header.getRPCBXMin());
        bx_max = std::min(bx_max, header.getRPCBXMax());
        LogDebug("RPCTwinMuxRawToDigi") << "BX range set to " << bx_min << ", " << bx_max;
    }

    bool has_first_rpc_word(false);
    rpctwinmux::RPCRecord rpc_record;
    for ( ; size > 1 ; --size, ++word) {
        if (calculate_crc_) {
            compute_crc_64bit(crc, *word);
        }
        unsigned int type(rpctwinmux::TwinMuxRecord::getType(*word));
        LogDebug("RPCTwinMuxRawToDigi") << "TwinMux data type " << std::hex << type << std::dec;
        if (type == rpctwinmux::TwinMuxRecord::rpc_first_type_) {
            if (has_first_rpc_word) {
                processRPCRecord(fed, amc_number, bx_counter, rpc_record, counters, digis, bx_min, bx_max, 0, 1);
            }
            rpc_record.reset();
            rpc_record.set(0, *word);
            has_first_rpc_word = true;
        } else if (type == rpctwinmux::TwinMuxRecord::rpc_second_type_) {
            if (!has_first_rpc_word) {
                edm::LogWarning("RPCTwinMuxRawToDigi") << "Received second RPC word without first";
            } else {
                rpc_record.set(1, *word);
                processRPCRecord(fed, amc_number, bx_counter, rpc_record, counters, digis, bx_min, bx_max, 0, 4);
                has_first_rpc_word = false;
            }
        }
    }
    if (has_first_rpc_word) {
        processRPCRecord(fed, amc_number, bx_counter, rpc_record, counters, digis, bx_min, bx_max, 0, 1);
    }

    rpctwinmux::TwinMuxTrailer trailer(*word);
    LogDebug("RPCTwinMuxRawToDigi") << "TwinMux Trailer " << std::hex << *word << std::dec;
    if (calculate_crc_) {
        compute_crc_64bit(crc, *word);
    }
    ++word;
    return true;
}

void RPCTwinMuxRawToDigi::processRPCRecord(int fed, unsigned int amc_number
                                           , unsigned int bx_counter
                                           , rpctwinmux::RPCRecord const & record
                                           , RPCAMCLinkCounters & counters
                                           , std::set<std::pair<RPCDetId, RPCDigi> > & digis
                                           , int bx_min, int bx_max
                                           , unsigned int link, unsigned int link_max) const
{
    LogDebug("RPCTwinMuxRawToDigi") << "RPCRecord " << std::hex << record.getRecord()[0]
                                    << ", " << record.getRecord()[1] << std::dec << std::endl;
    int bx_offset(record.getBXOffset());
    RPCAMCLink tm_link(fed, amc_number);
    for ( ; link <= link_max ; ++link) {
        tm_link.setAMCInput(link);
        rpctwinmux::RPCLinkRecord link_record(record.getRPCLinkRecord(link));

        if (link_record.isError()) {
            if (fill_counters_ && bx_offset == 0) {
                counters.add(RPCAMCLinkEvents::input_link_error_, tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Link in error for " << tm_link;
            continue;
        } else if (!link_record.isAcknowledge()) {
            if (fill_counters_ && bx_offset == 0) {
                counters.add(RPCAMCLinkEvents::input_link_ack_fail_, tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Link without acknowledge for " << tm_link;
            continue;
        }

        if (!link_record.getPartitionData()) {
            continue;
        }

        int bx(bx_offset - (int)(link_record.getDelay()));
        LogDebug("RPCTwinMuxRawToDigi") << "RPC BX " << bx << " for offset " << bx_offset;

        if (fill_counters_ && bx == 0 && link_record.isEOD()) { // EOD comes at the last delay
            counters.add(RPCAMCLinkEvents::input_eod_, tm_link);
        }

        RPCAMCLinkMap::map_type::const_iterator tm_link_it = es_tm_link_map_->getMap().find(tm_link);
        if (tm_link_it == es_tm_link_map_->getMap().end()) {
            if (fill_counters_ && bx_offset == 0) {
                counters.add(RPCAMCLinkEvents::amc_link_invalid_, RPCAMCLink(fed, amc_number));
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Skipping unknown TwinMuxLink " << tm_link;
            continue;
        }

        RPCLBLink lb_link = tm_link_it->second;

        if (link_record.getLinkBoard() > (unsigned int)RPCLBLink::max_linkboard_) {
            if (fill_counters_ && bx_offset == 0) {
                counters.add(RPCAMCLinkEvents::input_lb_invalid_, tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Skipping invalid LinkBoard " << link_record.getLinkBoard()
                                            << " for record " << link << " (" << std::hex << link_record.getRecord()
                                            << " in " << record.getRecord()[0] << ':' << record.getRecord()[1] << std::dec
                                            << " from " << tm_link;
            continue;
        }

        if (link_record.getConnector() > (unsigned int)RPCLBLink::max_connector_) {
            if (fill_counters_ && bx_offset == 0) {
                counters.add(RPCAMCLinkEvents::input_connector_invalid_, tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Skipping invalid Connector " << link_record.getConnector()
                                            << " for record " << link << " (" << std::hex << link_record.getRecord()
                                            << " in " << record.getRecord()[0] << ':' << record.getRecord()[1] << std::dec
                                            << ") from " << tm_link;
            continue;
        }

        lb_link.setLinkBoard(link_record.getLinkBoard());
        lb_link.setConnector(link_record.getConnector());

        RPCLBLinkMap::map_type::const_iterator lb_link_it = es_lb_link_map_->getMap().find(lb_link);
        if (lb_link_it == es_lb_link_map_->getMap().end()) {
            if (fill_counters_ && bx_offset == 0) {
                counters.add(RPCAMCLinkEvents::input_connector_not_used_, tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Could not find " << lb_link
                                            << " for record " << link << " (" << std::hex << link_record.getRecord()
                                            << " in " << record.getRecord()[0] << ':' << record.getRecord()[1] << std::dec
                                            << ") from " << tm_link;
            continue;
        }

        if (bx < bx_min || bx > bx_max) {
            continue;
        }

        if (fill_counters_ && bx == 0) {
            counters.add(RPCAMCLinkEvents::amc_event_, RPCAMCLink(fed, amc_number));
            counters.add(RPCAMCLinkEvents::input_event_, tm_link);
        }

        RPCFebConnector const & feb_connector(lb_link_it->second);
        RPCDetId det_id(feb_connector.getRPCDetId());
        unsigned int channel_offset(link_record.getPartition() ? 9 : 1); // 1-16
        std::uint8_t data(link_record.getPartitionData());

        for (unsigned int channel = 0 ; channel < 8 ; ++channel) {
            if (data & (0x1 << channel)) {
                unsigned int strip(feb_connector.getStrip(channel + channel_offset));
                if (strip) {
                    digis.insert(std::pair<RPCDetId, RPCDigi>(det_id, RPCDigi(strip, bx)));
                    LogDebug("RPCTwinMuxRawToDigi") << "RPCDigi " << det_id.rawId()
                                                    << ", " << strip << ", " << bx;
                }
            }
        }

        // rpctwinmux::RPCBXRecord checks postponed: not implemented in firmware as planned and tbd if design or firmware should change

    }
}

void RPCTwinMuxRawToDigi::putRPCDigis(edm::Event & event
                                      , std::set<std::pair<RPCDetId, RPCDigi> > const & digis)
{
    std::unique_ptr<RPCDigiCollection> rpc_digi_collection(new RPCDigiCollection());
    RPCDetId rpc_det_id;
    std::vector<RPCDigi> local_digis;
    for (std::pair<RPCDetId, RPCDigi> const & rpc_digi : digis) {
        LogDebug("RPCTwinMuxRawToDigi") << "RPCDigi " << rpc_digi.first.rawId()
                                        << ", " << rpc_digi.second.strip() << ", " << rpc_digi.second.bx();
        if (rpc_digi.first != rpc_det_id) {
            if (!local_digis.empty()) {
                rpc_digi_collection->put(RPCDigiCollection::Range(local_digis.begin(), local_digis.end()), rpc_det_id);
                local_digis.clear();
            }
            rpc_det_id = rpc_digi.first;
        }
        local_digis.push_back(rpc_digi.second);
    }
    if (!local_digis.empty()) {
        rpc_digi_collection->put(RPCDigiCollection::Range(local_digis.begin(), local_digis.end()), rpc_det_id);
    }

    event.put(std::move(rpc_digi_collection));
}

void RPCTwinMuxRawToDigi::putCounters(edm::Event & event
                                      , std::unique_ptr<RPCAMCLinkCounters>counters)
{
    event.put(std::move(counters));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCTwinMuxRawToDigi);

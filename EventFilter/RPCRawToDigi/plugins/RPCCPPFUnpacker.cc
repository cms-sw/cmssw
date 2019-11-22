#include "EventFilter/RPCRawToDigi/plugins/RPCCPPFUnpacker.h"

#include <algorithm>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/DataRecord/interface/RPCLBLinkMapRcd.h"
#include "DataFormats/RPCDigi/interface/RPCAMCLinkCounters.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCAMCLinkEvents.h"
#include "EventFilter/RPCRawToDigi/interface/RPCMP7Record.h"

RPCCPPFUnpacker::RPCCPPFUnpacker(edm::ParameterSet const& config, edm::ProducesCollector producesCollector)
    : RPCAMCUnpacker(config, producesCollector),
      fill_counters_(config.getParameter<bool>("fillAMCCounters")),
      bx_min_(config.getParameter<int>("bxMin")),
      bx_max_(config.getParameter<int>("bxMax")) {
  producesCollector.produces<RPCDigiCollection>();
  producesCollector.produces<l1t::CPPFDigiCollection>();
  if (fill_counters_) {
    producesCollector.produces<RPCAMCLinkCounters>("RPCAMCUnpacker");
  }
}

void RPCCPPFUnpacker::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  if (es_cppf_link_map_watcher_.check(setup)) {
    setup.get<RPCCPPFLinkMapRcd>().get(es_cppf_link_map_);
    std::set<int> feds;
    for (auto const& cppf_link : es_cppf_link_map_->getMap()) {
      feds.insert(cppf_link.first.getFED());
    }
    feds_.assign(feds.begin(), feds.end());
  }
}

void RPCCPPFUnpacker::produce(edm::Event& event,
                              edm::EventSetup const& setup,
                              std::map<RPCAMCLink, rpcamc13::AMCPayload> const& amc_payload) {
  // Get EventSetup Electronics Maps
  setup.get<RPCCPPFLinkMapRcd>().get(es_cppf_link_map_);
  setup.get<RPCLBLinkMapRcd>().get(es_lb_link_map_);

  std::set<std::pair<RPCDetId, RPCDigi> > rpc_digis;
  std::unique_ptr<l1t::CPPFDigiCollection> rpc_cppf_digis(new l1t::CPPFDigiCollection());
  std::unique_ptr<RPCAMCLinkCounters> counters(new RPCAMCLinkCounters());

  for (std::pair<RPCAMCLink const, rpcamc13::AMCPayload> const& payload : amc_payload) {
    processCPPF(payload.first, payload.second, *counters, rpc_digis, *rpc_cppf_digis);
  }

  putRPCDigis(event, rpc_digis);
  std::sort(rpc_cppf_digis->begin(), rpc_cppf_digis->end());
  event.put(std::move(rpc_cppf_digis));
  if (fill_counters_) {
    event.put(std::move(counters), "RPCAMCUnpacker");
  }
}

bool RPCCPPFUnpacker::processCPPF(RPCAMCLink const& link,
                                  rpcamc13::AMCPayload const& payload,
                                  RPCAMCLinkCounters& counters,
                                  std::set<std::pair<RPCDetId, RPCDigi> >& rpc_digis,
                                  l1t::CPPFDigiCollection& rpc_cppf_digis) const {
  LogDebug("RPCCPPFRawToDigi") << "RPCCPPF " << link << ", size " << payload.getData().size();

  if (!payload.isValid()) {
    return false;
  }

  std::vector<std::uint64_t>::const_iterator word(payload.getData().begin());
  rpcmp7::Header header(&(*word));
  ++word;
  ++word;
  rpcmp7::SubHeader subheader(*word);
  ++word;
  std::vector<std::uint64_t>::const_iterator word_end(payload.getData().end());
  --word_end;  // rpcamc::Trailer
  unsigned int bx_counter(header.getBXCounter());
  unsigned int bx_counter_mod(bx_counter % 27);

  int bx_min(bx_min_), bx_max(bx_max_);
  // no adjustable bx window implemented in readout yet

  unsigned int pos(0), length(0);
  unsigned int caption_id(0);
  bool zs_per_bx(false), have_bx_header(false);
  unsigned int bx_words(0);
  unsigned int block_id(0);
  std::uint32_t records[2];
  for (; word != word_end; ++word) {
    records[0] = *word & 0xffffffff;
    records[1] = (*word >> 32) & 0xffffffff;
    for (std::uint32_t* record = &(records[0]); record < &(records[2]); ++record) {
      if (pos >= length) {
        rpcmp7::BlockHeader block_header(*record);
        LogDebug("RPCCPPFUnpacker") << "block header " << std::hex << block_header.getRecord() << std::dec
                                    << " caption id " << block_header.getCaptionId() << " zs per bx "
                                    << block_header.hasZeroSuppressionPerBX() << " zs "
                                    << block_header.isZeroSuppressed() << " length " << block_header.getLength() << " "
                                    << word_end - word;
        pos = 0;
        length = block_header.getLength();
        caption_id = block_header.getCaptionId();
        zs_per_bx = block_header.hasZeroSuppressionPerBX();
        have_bx_header = false;
        bx_words = 6;
        block_id = block_header.getId();
      } else if (zs_per_bx && !have_bx_header) {
        // rpcmp7::BXHeader bx_header(*record);
        have_bx_header = true;
      } else {
        if (caption_id == 0x01) {  // RX
          processRXRecord(link, bx_counter_mod, rpccppf::RXRecord(*record), counters, rpc_digis, bx_min, bx_max);
        } else if (caption_id == 0x02) {  // TX
          processTXRecord(link, block_id, 6 - bx_words, rpccppf::TXRecord(*record), rpc_cppf_digis);
        }
        ++pos;
        --bx_words;
      }
    }
  }

  return true;
}

void RPCCPPFUnpacker::processRXRecord(RPCAMCLink link,
                                      unsigned int bx_counter_mod,
                                      rpccppf::RXRecord const& record,
                                      RPCAMCLinkCounters& counters,
                                      std::set<std::pair<RPCDetId, RPCDigi> >& rpc_digis,
                                      int bx_min,
                                      int bx_max) const {
  LogDebug("RPCCPPFRawToDigi") << "RXRecord " << std::hex << record.getRecord() << std::dec << std::endl;
  unsigned int fed(link.getFED());
  unsigned int amc_number(link.getAMCNumber());
  link.setAMCInput(record.getLink());

  int bx_offset = (int)(record.getBXCounterMod() + 31 - bx_counter_mod) % 27 - 4;

  if (record.isError()) {
    if (fill_counters_ && bx_offset == 0) {
      counters.add(RPCAMCLinkEvents::input_link_error_, link);
    }
    LogDebug("RPCCPPFRawToDigi") << "Link in error for " << link;
    return;
  } else if (!record.isAcknowledge()) {
    if (fill_counters_ && bx_offset == 0) {
      counters.add(RPCAMCLinkEvents::input_link_ack_fail_, link);
    }
    LogDebug("RPCCPPFRawToDigi") << "Link without acknowledge for " << link;
    return;
  }

  std::uint8_t data(record.getPartitionData());
  if (!data) {
    return;
  }

  int bx(bx_offset - (int)(record.getDelay()));
  LogDebug("RPCCPPFRawToDigi") << "RPC BX " << bx << " for offset " << bx_offset;

  if (fill_counters_ && bx == 0 && record.isEOD()) {  // EOD comes at the last delay
    counters.add(RPCAMCLinkEvents::input_eod_, link);
  }

  RPCAMCLinkMap::map_type::const_iterator link_it = es_cppf_link_map_->getMap().find(link);
  if (link_it == es_cppf_link_map_->getMap().end()) {
    if (fill_counters_ && bx_offset == 0) {
      counters.add(RPCAMCLinkEvents::amc_link_invalid_, RPCAMCLink(fed, amc_number));
    }
    LogDebug("RPCCPPFRawToDigi") << "Skipping unknown RPCCPPFLink " << link;
    return;
  }

  RPCLBLink lb_link = link_it->second;

  if (record.getLinkBoard() > (unsigned int)RPCLBLink::max_linkboard_) {
    if (fill_counters_ && bx_offset == 0) {
      counters.add(RPCAMCLinkEvents::input_lb_invalid_, link);
    }
    LogDebug("RPCCPPFRawToDigi") << "Skipping invalid LinkBoard " << record.getLinkBoard() << " for record " << link
                                 << " (" << std::hex << record.getRecord() << " in " << record.getRecord() << std::dec
                                 << " from " << link;
    return;
  }

  if (record.getConnector() > (unsigned int)RPCLBLink::max_connector_) {
    if (fill_counters_ && bx_offset == 0) {
      counters.add(RPCAMCLinkEvents::input_connector_invalid_, link);
    }
    LogDebug("RPCCPPFRawToDigi") << "Skipping invalid Connector " << record.getConnector() << " for record " << link
                                 << " (" << std::hex << record.getRecord() << " in " << record.getRecord() << std::dec
                                 << ") from " << link;
    return;
  }

  lb_link.setLinkBoard(record.getLinkBoard());
  lb_link.setConnector(record.getConnector());

  RPCLBLinkMap::map_type::const_iterator lb_link_it = es_lb_link_map_->getMap().find(lb_link);
  if (lb_link_it == es_lb_link_map_->getMap().end()) {
    if (fill_counters_ && bx_offset == 0) {
      counters.add(RPCAMCLinkEvents::input_connector_not_used_, link);
    }
    LogDebug("RPCCPPFRawToDigi") << "Could not find " << lb_link << " for record " << link << " (" << std::hex
                                 << record.getRecord() << " in " << record.getRecord() << std::dec << ") from " << link;
    return;
  }

  if (bx < bx_min || bx > bx_max) {
    return;
  }

  if (fill_counters_ && bx == 0) {
    counters.add(RPCAMCLinkEvents::input_event_, link);
  }

  RPCFebConnector const& feb_connector(lb_link_it->second);
  RPCDetId det_id(feb_connector.getRPCDetId());
  unsigned int channel_offset(record.getPartition() ? 9 : 1);  // 1-16

  for (unsigned int channel = 0; channel < 8; ++channel) {
    if (data & (0x1 << channel)) {
      unsigned int strip(feb_connector.getStrip(channel + channel_offset));
      if (strip) {
        rpc_digis.insert(std::pair<RPCDetId, RPCDigi>(det_id, RPCDigi(strip, bx)));
        LogDebug("RPCCPPFRawToDigi") << "RPCDigi " << det_id.rawId() << ", " << strip << ", " << bx;
      }
    }
  }
}

void RPCCPPFUnpacker::processTXRecord(RPCAMCLink link,
                                      unsigned int block,
                                      unsigned int word,
                                      rpccppf::TXRecord const& record,
                                      l1t::CPPFDigiCollection& rpc_cppf_digis) const {
  if (!record.isValid(0) && !record.isValid(1)) {
    return;
  }

  // This translation should become part of a CondFormat
  if (word > 5) {
    return;
  }
  static int const ring[6] = {2, 2, 2, 3, 2, 3};
  static int const station[6] = {1, 2, 3, 3, 4, 4};
  int region(link.getAMCNumber() < 7 ? 1 : -1);
  unsigned int endcap_sector((35 + (link.getAMCNumber() - (region > 0 ? 3 : 7)) * 9 + (block >> 1)) % 36 + 1);
  RPCDetId rpc_id(region,
                  ring[word]  // ring
                  ,
                  station[word]  // station
                  ,
                  (endcap_sector / 6) + 1  // sector
                  ,
                  1  // layer
                  ,
                  (endcap_sector - 1) % 6 + 1  // subsector
                  ,
                  0);  // roll

  if (record.isValid(0)) {
    rpc_cppf_digis.push_back(l1t::CPPFDigi(rpc_id, 0, record.getTheta(0), record.getPhi(0)));
  }
  if (record.isValid(1)) {
    rpc_cppf_digis.push_back(l1t::CPPFDigi(rpc_id, 0, record.getTheta(1), record.getPhi(1)));
  }
}

void RPCCPPFUnpacker::putRPCDigis(edm::Event& event, std::set<std::pair<RPCDetId, RPCDigi> > const& rpc_digis) const {
  std::unique_ptr<RPCDigiCollection> rpc_digi_collection(new RPCDigiCollection());
  RPCDetId rpc_det_id;
  std::vector<RPCDigi> local_rpc_digis;
  for (std::pair<RPCDetId, RPCDigi> const& rpc_digi : rpc_digis) {
    LogDebug("RPCCPPFRawToDigi") << "RPCDigi " << rpc_digi.first.rawId() << ", " << rpc_digi.second.strip() << ", "
                                 << rpc_digi.second.bx();
    if (rpc_digi.first != rpc_det_id) {
      if (!local_rpc_digis.empty()) {
        rpc_digi_collection->put(RPCDigiCollection::Range(local_rpc_digis.begin(), local_rpc_digis.end()), rpc_det_id);
        local_rpc_digis.clear();
      }
      rpc_det_id = rpc_digi.first;
    }
    local_rpc_digis.push_back(rpc_digi.second);
  }
  if (!local_rpc_digis.empty()) {
    rpc_digi_collection->put(RPCDigiCollection::Range(local_rpc_digis.begin(), local_rpc_digis.end()), rpc_det_id);
  }

  event.put(std::move(rpc_digi_collection));
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "EventFilter/RPCRawToDigi/plugins/RPCAMCUnpackerFactory.h"
DEFINE_EDM_PLUGIN(RPCAMCUnpackerFactory, RPCCPPFUnpacker, "RPCCPPFUnpacker");

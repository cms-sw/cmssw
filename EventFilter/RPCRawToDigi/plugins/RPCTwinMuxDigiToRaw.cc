#include "EventFilter/RPCRawToDigi/plugins/RPCTwinMuxDigiToRaw.h"

#include <cstdint>
#include <cstring>
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

#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxPacker.h"
#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

RPCTwinMuxDigiToRaw::RPCTwinMuxDigiToRaw(edm::ParameterSet const& config)
    : es_tm_link_map_br_token_(esConsumes<RPCAMCLinkMap, RPCTwinMuxLinkMapRcd, edm::Transition::BeginRun>()),
      es_tm_link_map_token_(esConsumes<RPCInverseAMCLinkMap, RPCInverseTwinMuxLinkMapRcd>()),
      es_lb_link_map_token_(esConsumes<RPCInverseLBLinkMap, RPCInverseLBLinkMapRcd>()),
      bx_min_(config.getParameter<int>("bxMin")),
      bx_max_(config.getParameter<int>("bxMax")),
      ignore_eod_(config.getParameter<bool>("ignoreEOD")),
      event_type_(config.getParameter<int>("eventType")),
      ufov_(config.getParameter<unsigned int>("uFOV")) {
  produces<FEDRawDataCollection>();
  digi_token_ = consumes<RPCDigiCollection>(config.getParameter<edm::InputTag>("inputTag"));
}

RPCTwinMuxDigiToRaw::~RPCTwinMuxDigiToRaw() {}

void RPCTwinMuxDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions& descs) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag", edm::InputTag("simMuonRPCDigis", ""));
  desc.add<int>("bxMin", -2);
  desc.add<int>("bxMax", 2);
  desc.add<bool>("ignoreEOD", true);
  desc.add<int>("eventType", 1);
  desc.add<unsigned int>("uFOV", 1);
  descs.add("RPCTwinMuxDigiToRaw", desc);
}

void RPCTwinMuxDigiToRaw::beginRun(edm::Run const& run, edm::EventSetup const& setup) {
  if (es_tm_link_map_watcher_.check(setup)) {
    edm::ESHandle<RPCAMCLinkMap> es_tm_link_map = setup.getHandle(es_tm_link_map_br_token_);
    fed_amcs_.clear();
    for (auto const& tm_link : es_tm_link_map->getMap()) {
      RPCAMCLink amc(tm_link.first);
      amc.setAMCInput();
      fed_amcs_[amc.getFED()].push_back(amc);
    }
    for (auto& fed_amcs : fed_amcs_) {
      std::sort(fed_amcs.second.begin(), fed_amcs.second.end());
      fed_amcs.second.erase(std::unique(fed_amcs.second.begin(), fed_amcs.second.end()), fed_amcs.second.end());
    }
  }
}

void RPCTwinMuxDigiToRaw::produce(edm::Event& event, edm::EventSetup const& setup) {
  // Get EventSetup Electronics Maps
  edm::ESHandle<RPCInverseAMCLinkMap> es_tm_link_map_ = setup.getHandle(es_tm_link_map_token_);
  edm::ESHandle<RPCInverseLBLinkMap> es_lb_link_map = setup.getHandle(es_lb_link_map_token_);

  // Get Digi Collection
  edm::Handle<RPCDigiCollection> digi_collection;
  event.getByToken(digi_token_, digi_collection);

  // Create output
  std::unique_ptr<FEDRawDataCollection> data_collection(new FEDRawDataCollection());

  std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > > amc_bx_tmrecord;
  RPCTwinMuxPacker::getRPCTwinMuxRecords(*es_lb_link_map,
                                         *es_tm_link_map_,
                                         bx_min_,
                                         bx_max_,
                                         event.bunchCrossing(),
                                         *digi_collection,
                                         amc_bx_tmrecord,
                                         ignore_eod_);

  std::map<int, FEDRawData> fed_data;
  // Loop over the FEDs
  for (std::pair<const int, std::vector<RPCAMCLink> > const& fed_amcs : fed_amcs_) {
    FEDRawData& data = data_collection->FEDData(fed_amcs.first);
    unsigned int size(0);

    // FED Header + BLOCK Header (1 word + 1 word)
    data.resize((size + 2) * 8);
    // FED Header
    FEDHeader::set(data.data() + size * 8, event_type_, event.id().event(), event.bunchCrossing(), fed_amcs.first);
    ++size;
    // BLOCK Header
    rpctwinmux::BlockHeader block_header(ufov_, fed_amcs.second.size(), event.eventAuxiliary().orbitNumber());
    std::memcpy(data.data() + size * 8, &block_header.getRecord(), 8);
    ++size;

    // BLOCK AMC Content - 1 word each
    data.resize((size + fed_amcs.second.size()) * 8);
    unsigned int block_content_size(0);
    for (RPCAMCLink const& amc : fed_amcs.second) {
      std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > >::const_iterator bx_tmrecord(
          amc_bx_tmrecord.find(amc));
      unsigned int block_amc_size(3 + 2 * (bx_tmrecord == amc_bx_tmrecord.end() ? 0 : bx_tmrecord->second.size()));
      block_content_size += block_amc_size;
      rpctwinmux::BlockAMCContent amc_content(
          true, true, true, true, true, true, true, block_amc_size, 0, amc.getAMCNumber(), 0);
      std::memcpy(data.data() + size * 8, &amc_content.getRecord(), 8);
      ++size;
    }

    // AMC Payload - 2 words header, 1 word trailer, 2 words per RPCRecord
    data.resize((size + block_content_size) * 8);
    for (RPCAMCLink const& amc : fed_amcs.second) {
      // TwinMux Header
      std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > >::const_iterator bx_tmrecord(
          amc_bx_tmrecord.find(amc));
      unsigned int block_amc_size(3 + 2 * (bx_tmrecord == amc_bx_tmrecord.end() ? 0 : bx_tmrecord->second.size()));

      rpctwinmux::TwinMuxHeader tm_header(amc.getAMCNumber(),
                                          event.id().event(),
                                          event.bunchCrossing(),
                                          block_amc_size,
                                          event.eventAuxiliary().orbitNumber(),
                                          0);
      tm_header.setRPCBXWindow(bx_min_, bx_min_);
      std::memcpy(data.data() + size * 8, tm_header.getRecord(), 16);
      size += 2;

      if (bx_tmrecord != amc_bx_tmrecord.end()) {
        for (std::vector<std::pair<int, rpctwinmux::RPCRecord> >::const_iterator tmrecord = bx_tmrecord->second.begin();
             tmrecord != bx_tmrecord->second.end();
             ++tmrecord) {
          std::memcpy(data.data() + size * 8, tmrecord->second.getRecord(), 16);
          size += 2;
        }
      }

      rpctwinmux::TwinMuxTrailer tm_trailer(0x0, event.id().event(), 3 + 2 * block_amc_size);
      std::memcpy(data.data() + size * 8, &tm_trailer.getRecord(), 8);
      ++size;
      // CRC32 not calculated (for now)
    }

    // BLOCK Trailer + FED Trailer (1 word + 1 word)
    data.resize((size + 2) * 8);
    // BLOCK Trailer
    rpctwinmux::BlockTrailer block_trailer(0x0, 0, event.id().event(), event.bunchCrossing());
    std::memcpy(data.data() + size * 8, &block_trailer.getRecord(), 8);
    ++size;
    // CRC32 not calculated (for now)

    // FED Trailer
    ++size;
    FEDTrailer::set(data.data() + (size - 1) * 8, size, 0x0, 0, 0);
    std::uint16_t crc(evf::compute_crc(data.data(), size * 8));
    FEDTrailer::set(data.data() + (size - 1) * 8, size, crc, 0, 0);
  }

  event.put(std::move(data_collection));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCTwinMuxDigiToRaw);

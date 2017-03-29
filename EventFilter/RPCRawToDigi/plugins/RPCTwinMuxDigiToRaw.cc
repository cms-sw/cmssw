#include "EventFilter/RPCRawToDigi/plugins/RPCTwinMuxDigiToRaw.h"

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

#include "CondFormats/DataRecord/interface/RPCInverseLBLinkMapRcd.h"
#include "CondFormats/DataRecord/interface/RPCInverseTwinMuxLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCInverseLBLinkMap.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxPacker.h"
#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

RPCTwinMuxDigiToRaw::RPCTwinMuxDigiToRaw(edm::ParameterSet const & _config)
    : bx_min_(_config.getParameter<int>("bxMin"))
    , bx_max_(_config.getParameter<int>("bxMax"))
    , ignore_eod_(_config.getParameter<bool>("ignoreEOD"))
    , event_type_(_config.getParameter<int>("eventType"))
    , ufov_(_config.getParameter<unsigned int>("uFOV"))
{
    produces<FEDRawDataCollection>();
    digi_token_ = consumes<RPCDigiCollection>(_config.getParameter<edm::InputTag>("inputTag"));
}

RPCTwinMuxDigiToRaw::~RPCTwinMuxDigiToRaw()
{}

void RPCTwinMuxDigiToRaw::fillDescriptions(edm::ConfigurationDescriptions & _descs)
{
    edm::ParameterSetDescription _desc;
    _desc.add<edm::InputTag>("inputTag", edm::InputTag("simMuonRPCDigis", ""));
    _desc.add<int>("bxMin", -2);
    _desc.add<int>("bxMax", 2);
    _desc.add<bool>("ignoreEOD", true);
    _desc.add<int>("eventType", 1);
    _desc.add<unsigned int>("uFOV", 1);
    _descs.add("RPCTwinMuxDigiToRaw", _desc);
}

void RPCTwinMuxDigiToRaw::beginRun(edm::Run const & _run, edm::EventSetup const & _setup)
{
    if (es_tm_link_map_watcher_.check(_setup)) {
        edm::ESHandle<RPCAMCLinkMap> _es_tm_link_map;
        _setup.get<RPCTwinMuxLinkMapRcd>().get(_es_tm_link_map);
        fed_amcs_.clear();
        for (auto const & _tm_link : _es_tm_link_map->getMap()) {
            RPCAMCLink _amc(_tm_link.first);
            _amc.setAMCInput();
            fed_amcs_[_amc.getFED()].push_back(_amc);
        }
        for (auto & _fed_amcs : fed_amcs_) {
            std::sort(_fed_amcs.second.begin(), _fed_amcs.second.end());
            _fed_amcs.second.erase(std::unique(_fed_amcs.second.begin(), _fed_amcs.second.end()), _fed_amcs.second.end());
        }
    }
}

void RPCTwinMuxDigiToRaw::produce(edm::Event & _event, edm::EventSetup const & _setup)
{
    // Get EventSetup Electronics Maps
    edm::ESHandle<RPCInverseAMCLinkMap> _es_tm_link_map_;
    edm::ESHandle<RPCInverseLBLinkMap> _es_lb_link_map;

    _setup.get<RPCInverseTwinMuxLinkMapRcd>().get(_es_tm_link_map_);
    _setup.get<RPCInverseLBLinkMapRcd>().get(_es_lb_link_map);

    // Get Digi Collection
    edm::Handle<RPCDigiCollection> _digi_collection;
    _event.getByToken(digi_token_, _digi_collection);

    // Create output
    std::unique_ptr<FEDRawDataCollection> _data_collection(new FEDRawDataCollection());

    std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > > _amc_bx_tmrecord;
    RPCTwinMuxPacker::getRPCTwinMuxRecords(*_es_lb_link_map, *_es_tm_link_map_
                                           , bx_min_, bx_max_, _event.bunchCrossing()
                                           , *_digi_collection
                                           , _amc_bx_tmrecord
                                           , ignore_eod_);

    std::map<int, FEDRawData> _fed_data;
    // Loop over the FEDs
    for (std::pair<int, std::vector<RPCAMCLink> > const & _fed_amcs : fed_amcs_) {
        FEDRawData & _data = _data_collection->FEDData(_fed_amcs.first);
        unsigned int _size(0);

        // FED Header + BLOCK Header (1 word + 1 word)
        _data.resize((_size + 2) * 8);
        // FED Header
        FEDHeader::set(_data.data() + _size * 8, event_type_, _event.id().event(), _event.bunchCrossing(), _fed_amcs.first);
        ++_size;
        // BLOCK Header
        rpctwinmux::BlockHeader _block_header(ufov_, _fed_amcs.second.size(), _event.eventAuxiliary().orbitNumber());
        std::memcpy(_data.data() + _size * 8, &_block_header.getRecord(), 8);
        ++_size;

        // BLOCK AMC Content - 1 word each
        _data.resize((_size + _fed_amcs.second.size()) * 8);
        unsigned int _block_content_size(0);
        for (RPCAMCLink const & _amc : _fed_amcs.second) {
            std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > >::const_iterator _bx_tmrecord(_amc_bx_tmrecord.find(_amc));
            unsigned int _block_amc_size(3 + 2 * (_bx_tmrecord == _amc_bx_tmrecord.end() ? 0 : _bx_tmrecord->second.size()));
            _block_content_size += _block_amc_size;
            rpctwinmux::BlockAMCContent _amc_content(true
                                                     , true, true
                                                     , true, true, true, true, _block_amc_size
                                                     , 0, _amc.getAMCNumber()
                                                     , 0);
            std::memcpy(_data.data() + _size * 8, &_amc_content.getRecord(), 8);
            ++_size;
        }

        // AMC Payload - 2 words header, 1 word trailer, 2 words per RPCRecord
        _data.resize((_size + _block_content_size) * 8);
        for (RPCAMCLink const & _amc : _fed_amcs.second) {
            // TwinMux Header
            std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > >::const_iterator _bx_tmrecord(_amc_bx_tmrecord.find(_amc));
            unsigned int _block_amc_size(3 + 2 * (_bx_tmrecord == _amc_bx_tmrecord.end() ? 0 : _bx_tmrecord->second.size()));

            rpctwinmux::TwinMuxHeader _tm_header(_amc.getAMCNumber(), _event.id().event(), _event.bunchCrossing()
                                                 , _block_amc_size
                                                 , _event.eventAuxiliary().orbitNumber()
                                                 , 0);
            _tm_header.setRPCBXWindow(bx_min_, bx_min_);
            std::memcpy(_data.data() + _size * 8, _tm_header.getRecord(), 16);
            _size += 2;

            if (_bx_tmrecord != _amc_bx_tmrecord.end()) {
                for (std::vector<std::pair<int, rpctwinmux::RPCRecord> >::const_iterator _tmrecord = _bx_tmrecord->second.begin()
                         ; _tmrecord != _bx_tmrecord->second.end() ; ++_tmrecord) {
                    std::memcpy(_data.data() + _size * 8, _tmrecord->second.getRecord(), 16);
                    _size += 2;
                }
            }

            rpctwinmux::TwinMuxTrailer _tm_trailer(0x0, _event.id().event(),  3 + 2 * _block_amc_size);
            std::memcpy(_data.data() + _size * 8, &_tm_trailer.getRecord(), 8);
            ++_size;
            // CRC32 not calculated (for now)
        }

        // BLOCK Trailer + FED Trailer (1 word + 1 word)
        _data.resize((_size + 2) * 8);
        // BLOCK Trailer
        rpctwinmux::BlockTrailer _block_trailer(0x0, 0, _event.id().event(), _event.bunchCrossing());
        std::memcpy(_data.data() + _size * 8, &_block_trailer.getRecord(), 8);
        ++_size;
        // CRC32 not calculated (for now)

        // FED Trailer
        ++_size;
        FEDTrailer::set(_data.data() + (_size - 1) * 8, _size, 0x0, 0, 0);
        ::uint16_t _crc(evf::compute_crc(_data.data(), _size * 8));
        FEDTrailer::set(_data.data() + (_size - 1) * 8, _size, _crc, 0, 0);
    }

    _event.put(std::move(_data_collection));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCTwinMuxDigiToRaw);

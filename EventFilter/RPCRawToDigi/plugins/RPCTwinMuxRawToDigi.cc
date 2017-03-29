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

RPCTwinMuxRawToDigi::RPCTwinMuxRawToDigi(edm::ParameterSet const & _config)
    : calculate_crc_(_config.getParameter<bool>("calculateCRC"))
    , fill_counters_(_config.getParameter<bool>("fillCounters"))
    , bx_min_(_config.getParameter<int>("bxMin"))
    , bx_max_(_config.getParameter<int>("bxMax"))
{
    produces<RPCDigiCollection>();
    if (fill_counters_) {
        produces<RPCAMCLinkCounters>();
    }
    raw_token_ = consumes<FEDRawDataCollection>(_config.getParameter<edm::InputTag>("inputTag"));
}

RPCTwinMuxRawToDigi::~RPCTwinMuxRawToDigi()
{}

void RPCTwinMuxRawToDigi::compute_crc_64bit(::uint16_t & _crc, ::uint64_t const & _word)
{ // overcome constness problem evf::compute_crc_64bit
    unsigned char const * _uchars(reinterpret_cast<unsigned char const *>(&_word));
    for (unsigned char const * _uchar = _uchars + 7
             ; _uchar >= _uchars ; --_uchar) {
        _crc = evf::compute_crc_8bit(_crc, *_uchar);
    }
}

void RPCTwinMuxRawToDigi::fillDescriptions(edm::ConfigurationDescriptions & _descs)
{
    edm::ParameterSetDescription _desc;
    _desc.add<edm::InputTag>("inputTag", edm::InputTag("rawDataCollector", ""));
    _desc.add<bool>("calculateCRC", true);
    _desc.add<bool>("fillCounters", true);
    _desc.add<int>("bxMin", -2);
    _desc.add<int>("bxMax", 2);
    _descs.add("RPCTwinMuxRawToDigi", _desc);
}

void RPCTwinMuxRawToDigi::beginRun(edm::Run const & _run, edm::EventSetup const & _setup)
{
    if (es_tm_link_map_watcher_.check(_setup)) {
        _setup.get<RPCTwinMuxLinkMapRcd>().get(es_tm_link_map_);
        std::set<int> _feds;
        for (auto const & _tm_link : es_tm_link_map_->getMap()) {
            _feds.insert(_tm_link.first.getFED());
        }
        feds_.assign(_feds.begin(), _feds.end());
    }
}

void RPCTwinMuxRawToDigi::produce(edm::Event & _event, edm::EventSetup const & _setup)
{
    // Get EventSetup Electronics Maps
    _setup.get<RPCTwinMuxLinkMapRcd>().get(es_tm_link_map_);
    _setup.get<RPCLBLinkMapRcd>().get(es_lb_link_map_);

    // Get RAW Data
    edm::Handle<FEDRawDataCollection> _raw_data_collection;
    _event.getByToken(raw_token_, _raw_data_collection);

    std::set<std::pair<RPCDetId, RPCDigi> > _rpc_digis;
    std::unique_ptr<RPCAMCLinkCounters> _counters(new RPCAMCLinkCounters());

    // Loop over the FEDs
    for (int _fed : feds_) {

        if (fill_counters_) {
            _counters->add(RPCAMCLinkCounters::fed_event_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
        }

        ::uint16_t _crc(0xffff);

        FEDRawData const & _raw_data = _raw_data_collection->FEDData(_fed);
        unsigned int _nwords(_raw_data.size() / sizeof(::uint64_t));
        if (!_nwords) {
            continue;
        }

        ::uint64_t const * _word(reinterpret_cast<::uint64_t const *>(_raw_data.data()));
        ::uint64_t const * _word_end = _word + _nwords;

        LogDebug("RPCTwinMuxRawToDigi") << "Handling FED " << _fed << " with length " << _nwords;

        // Handle the CDF Headers
        if (!processCDFHeaders(_fed
                               , _word, _word_end
                               , _crc, *_counters)) {
            continue;
        }

        // Handle the CDF Trailers
        if (!processCDFTrailers(_fed, _nwords
                                , _word, _word_end
                                , _crc, *_counters)) {
            continue;
        }

        // Loop over the Blocks
        while (_word < _word_end) {
            processBlock(_fed
                         , _word, _word_end
                         , _crc, *_counters, _rpc_digis);
        }

        // Complete CRC check with trailer
        if (calculate_crc_) {
            _word = _word_end;
            _word_end = reinterpret_cast<::uint64_t const *>(_raw_data.data()) + _nwords - 1;
            for ( ; _word < _word_end ; ++_word) {
                compute_crc_64bit(_crc, *_word);
            }
            compute_crc_64bit(_crc, (*_word & 0xffffffff0000ffff)); // trailer excluding crc
            FEDTrailer _trailer(reinterpret_cast<unsigned char const *>(_word_end));
            if ((unsigned int)(_trailer.crc()) != _crc) {
                if (fill_counters_) {
                    _counters->add(RPCAMCLinkCounters::fed_trailer_crc_mismatch_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
                }
                edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Trailer CRC doesn't match for FED id " << _fed;
            }
        }
    }

    putRPCDigis(_event, _rpc_digis);
    if (fill_counters_) {
        putCounters(_event, std::move(_counters));
    }
}

bool RPCTwinMuxRawToDigi::processCDFHeaders(int _fed
                                            , ::uint64_t const * & _word, ::uint64_t const * & _word_end
                                            , ::uint16_t & _crc
                                            , RPCAMCLinkCounters & _counters) const
{
    bool _more_headers(true);
    for ( ; _word < _word_end && _more_headers ; ++_word) {
        if (calculate_crc_) {
            compute_crc_64bit(_crc, *_word);
        }

        LogDebug("RPCTwinMuxRawToDigi") << "CDF Header " << std::hex << *_word << std::dec;
        FEDHeader _header(reinterpret_cast<unsigned char const *>(_word));
        if (!_header.check()) {
            if (fill_counters_) {
                _counters.add(RPCAMCLinkCounters::fed_header_check_fail_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Header check failed for FED id " << _fed;
            ++_word;
            break;
        }
        if (_header.sourceID() != _fed) {
            if (fill_counters_) {
                _counters.add(RPCAMCLinkCounters::fed_header_id_mismatch_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Header Source ID " << _header.sourceID()
                                                   << " does not match requested FED id " << _fed;
            break;
        }

        // moreHeaders() not used
        // _more_headers = _header.moreHeaders();
        _more_headers = false;
    }

    return !_more_headers;
}

bool RPCTwinMuxRawToDigi::processCDFTrailers(int _fed, unsigned int _nwords
                                             , ::uint64_t const * & _word, ::uint64_t const * & _word_end
                                             , ::uint16_t & _crc
                                             , RPCAMCLinkCounters & _counters) const
{
    bool _more_trailers(true);
    for (--_word_end ; _word_end > _word && _more_trailers ; --_word_end) {
        FEDTrailer _trailer(reinterpret_cast<unsigned char const *>(_word_end));
        LogDebug("RPCTwinMuxRawToDigi") << "CDF Trailer " << std::hex << *_word_end << std::dec
                                        << ", length " << _trailer.lenght();
        if (!_trailer.check()) {
            if (fill_counters_) {
                _counters.add(RPCAMCLinkCounters::fed_trailer_check_fail_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Trailer check failed for FED id " << _fed;
            --_word_end;
            break;
        }
        if (_trailer.lenght() != (int)_nwords) {
            if (fill_counters_) {
                _counters.add(RPCAMCLinkCounters::fed_trailer_length_mismatch_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "FED Trailer length " << _trailer.lenght()
                                                   << " does not match actual data size " << _nwords
                                                   << " for FED id " << _fed;
            --_word_end;
            break;
        }

        // moreTrailers() not used
        // _more_trailers = _trailer.moreTrailers();
        _more_trailers = false;
    }

    ++_word_end;

    return !_more_trailers;
}

bool RPCTwinMuxRawToDigi::processBlock(int _fed
                                       , ::uint64_t const * & _word, ::uint64_t const * _word_end
                                       , ::uint16_t & _crc
                                       , RPCAMCLinkCounters & _counters
                                       , std::set<std::pair<RPCDetId, RPCDigi> > & _digis) const
{
    // Block Header and Content
    rpctwinmux::BlockHeader _block_header(*_word);
    if (calculate_crc_) {
        compute_crc_64bit(_crc, *_word);
    }
    ++_word;

    unsigned int _n_amc(_block_header.getNAMC());
    if (_word + _n_amc + 1 >= _word_end) {
        if (fill_counters_) {
            _counters.add(RPCAMCLinkCounters::fed_block_length_invalid_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "Block can not be complete for FED " << _fed;
        _word = _word_end;
        return false;
    }

    std::vector<std::pair<unsigned int, unsigned int> > _amc_size_map;
    for (unsigned int _amc = 0 ; _amc < _n_amc ; ++_amc) {
        LogDebug("RPCTwinMuxRawToDigi") << "Block AMC " << _amc;
        rpctwinmux::BlockAMCContent _amc_content(*_word);
        if (calculate_crc_) {
            compute_crc_64bit(_crc, *_word);
        }
        ++_word;

        _amc_size_map.push_back(std::make_pair(_amc_content.getAMCNumber(), _amc_content.getSize()));
        if (!_amc_content.isValid()) {
            if (fill_counters_) {
                _counters.add(RPCAMCLinkCounters::amc_evc_bc_invalid_, RPCAMCLink(_fed, _amc_content.getAMCNumber()));
            }
            edm::LogWarning("RPCTwinMuxRawToDigi") << "BlockAMCContent is reporting an invalid "
                                                   << "Event Counter or Bunch Counter for FED " << _fed
                                                   << ", AMC " << _amc_content.getAMCNumber();
        }
    }

    for (std::pair<unsigned int, unsigned int> const & _amc_size : _amc_size_map) {
        processTwinMux(_fed, _amc_size.first, _amc_size.second
                       , _word, _word_end
                       , _crc, _counters
                       , _digis);
    }

    if (_word < _word_end) {
        rpctwinmux::BlockTrailer _block_trailer(*_word);
        if (calculate_crc_) {
            compute_crc_64bit(_crc, *_word);
        }
        ++_word;
        return true;
    } else {
        return false;
    }
}

bool RPCTwinMuxRawToDigi::processTwinMux(int _fed, unsigned int _amc_number, unsigned int _size
                                         , ::uint64_t const * & _word, ::uint64_t const * _word_end
                                         , ::uint16_t & _crc
                                         , RPCAMCLinkCounters & _counters
                                         , std::set<std::pair<RPCDetId, RPCDigi> > & _digis) const
{
    LogDebug("RPCTwinMuxRawToDigi") << "TwinMux AMC#" << _amc_number << ", size " << _size;
    if (!_size) {
        return true;
    }
    if (_amc_number > (unsigned int)RPCAMCLink::max_amcnumber_) {
        if (fill_counters_) {
            _counters.add(RPCAMCLinkCounters::fed_block_amc_number_invalid_, RPCAMCLink(_fed, RPCAMCLink::wildcard_));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "Invalid AMC Number " << _amc_number
                                               << " for FED " << _fed;
        if (calculate_crc_) {
            for ( ; _size > 0 ; --_size, ++_word) {
                compute_crc_64bit(_crc, *_word);
            }
        } else {
            _word += _size;
        }
        return false;
    }
    if (_word + _size >= _word_end || _size < 3) {
        if (fill_counters_) {
            _counters.add(RPCAMCLinkCounters::amc_payload_length_invalid_, RPCAMCLink(_fed, _amc_number));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "TwinMux Data can not be complete for FED " << _fed << " AMC #" << _amc_number;
        if (calculate_crc_) {
            for ( ; _size > 0 ; --_size, ++_word) {
                compute_crc_64bit(_crc, *_word);
            }
        } else {
            _word += _size;
        }
        return false;
    }

    rpctwinmux::TwinMuxHeader _header(_word);
    unsigned int _bx_counter(_header.getBXCounter());
    if (calculate_crc_) {
        compute_crc_64bit(_crc, *_word); ++_word;
        compute_crc_64bit(_crc, *_word); ++_word;
    } else {
        _word += 2;
    }
    _size -= 2;

    if (_amc_number != _header.getAMCNumber()) {
        if (fill_counters_) {
            _counters.add(RPCAMCLinkCounters::amc_number_mismatch_, RPCAMCLink(_fed, _amc_number));
        }
        edm::LogWarning("RPCTwinMuxRawToDigi") << "AMC Number inconsistent in TwinMuxHeader vs BlockAMCContent: " << _header.getAMCNumber()
                                               << " vs " << _amc_number;
        if (calculate_crc_) {
            for ( ; _size > 0 ; --_size, ++_word) {
                compute_crc_64bit(_crc, *_word);
            }
        } else {
            _word += _size;
        }
        return false;
    }

    int _bx_min(bx_min_), _bx_max(bx_max_);
    if (_header.hasRPCBXWindow()) {
        _bx_min = std::max(_bx_min, _header.getRPCBXMin());
        _bx_max = std::min(_bx_max, _header.getRPCBXMax());
        LogDebug("RPCTwinMuxRawToDigi") << "BX range set to " << _bx_min << ", " << _bx_max;
    }

    bool _has_first_rpc_word(false);
    rpctwinmux::RPCRecord _rpc_record;
    for ( ; _size > 1 ; --_size, ++_word) {
        if (calculate_crc_) {
            compute_crc_64bit(_crc, *_word);
        }
        unsigned int _type(rpctwinmux::TwinMuxRecord::getType(*_word));
        LogDebug("RPCTwinMuxRawToDigi") << "TwinMux data type " << std::hex << _type << std::dec;
        if (_type == rpctwinmux::TwinMuxRecord::rpc_first_type_) {
            if (_has_first_rpc_word) {
                processRPCRecord(_fed, _amc_number, _bx_counter, _rpc_record, _counters, _digis, _bx_min, _bx_max, 0, 1);
            }
            _rpc_record.reset();
            _rpc_record.set(0, *_word);
            _has_first_rpc_word = true;
        } else if (_type == rpctwinmux::TwinMuxRecord::rpc_second_type_) {
            if (!_has_first_rpc_word) {
                edm::LogWarning("RPCTwinMuxRawToDigi") << "Received second RPC word without first";
            } else {
                _rpc_record.set(1, *_word);
                processRPCRecord(_fed, _amc_number, _bx_counter, _rpc_record, _counters, _digis, _bx_min, _bx_max, 0, 4);
                _has_first_rpc_word = false;
            }
        }
    }
    if (_has_first_rpc_word) {
        processRPCRecord(_fed, _amc_number, _bx_counter, _rpc_record, _counters, _digis, _bx_min, _bx_max, 0, 1);
    }

    rpctwinmux::TwinMuxTrailer _trailer(*_word);
    LogDebug("RPCTwinMuxRawToDigi") << "TwinMux Trailer " << std::hex << *_word << std::dec;
    if (calculate_crc_) {
        compute_crc_64bit(_crc, *_word);
    }
    ++_word;
    return true;
}

void RPCTwinMuxRawToDigi::processRPCRecord(int _fed, unsigned int _amc_number
                                           , unsigned int _bx_counter
                                           , rpctwinmux::RPCRecord const & _record
                                           , RPCAMCLinkCounters & _counters
                                           , std::set<std::pair<RPCDetId, RPCDigi> > & _digis
                                           , int _bx_min, int _bx_max
                                           , unsigned int _link, unsigned int _link_max) const
{
    LogDebug("RPCTwinMuxRawToDigi") << "RPCRecord " << std::hex << _record.getRecord()[0]
                                    << ", " << _record.getRecord()[1] << std::dec << std::endl;
    int _bx_offset(_record.getBXOffset());
    RPCAMCLink _tm_link(_fed, _amc_number);
    for ( ; _link <= _link_max ; ++_link) {
        _tm_link.setAMCInput(_link);
        rpctwinmux::RPCLinkRecord _link_record(_record.getRPCLinkRecord(_link));

        if (_link_record.isError()) {
            if (fill_counters_ && _bx_offset == 0) {
                _counters.add(RPCAMCLinkCounters::input_link_error_, _tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Link in error for " << _tm_link;
            continue;
        } else if (!_link_record.isAcknowledge()) {
            if (fill_counters_ && _bx_offset == 0) {
                _counters.add(RPCAMCLinkCounters::input_link_ack_fail_, _tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Link without acknowledge for " << _tm_link;
            continue;
        }

        if (!_link_record.getPartitionData()) {
            continue;
        }

        int _bx(_bx_offset - (int)(_link_record.getDelay()));
        LogDebug("RPCTwinMuxRawToDigi") << "RPC BX " << _bx << " for offset " << _bx_offset;

        if (fill_counters_ && _bx == 0 && _link_record.isEOD()) { // EOD comes at the last delay
            _counters.add(RPCAMCLinkCounters::input_eod_, _tm_link);
        }

        RPCAMCLinkMap::map_type::const_iterator _tm_link_it = es_tm_link_map_->getMap().find(_tm_link);
        if (_tm_link_it == es_tm_link_map_->getMap().end()) {
            if (fill_counters_ && _bx_offset == 0) {
                _counters.add(RPCAMCLinkCounters::amc_link_invalid_, RPCAMCLink(_fed, _amc_number));
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Skipping unknown TwinMuxLink " << _tm_link;
            continue;
        }

        RPCLBLink _lb_link = _tm_link_it->second;

        if (_link_record.getLinkBoard() > (unsigned int)RPCLBLink::max_linkboard_) {
            if (fill_counters_ && _bx_offset == 0) {
                _counters.add(RPCAMCLinkCounters::input_lb_invalid_, _tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Skipping invalid LinkBoard " << _link_record.getLinkBoard()
                                            << " for record " << _link << " (" << std::hex << _link_record.getRecord()
                                            << " in " << _record.getRecord()[0] << ':' << _record.getRecord()[1] << std::dec
                                            << " from " << _tm_link;
            continue;
        }

        if (_link_record.getConnector() > (unsigned int)RPCLBLink::max_connector_) {
            if (fill_counters_ && _bx_offset == 0) {
                _counters.add(RPCAMCLinkCounters::input_connector_invalid_, _tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Skipping invalid Connector " << _link_record.getConnector()
                                            << " for record " << _link << " (" << std::hex << _link_record.getRecord()
                                            << " in " << _record.getRecord()[0] << ':' << _record.getRecord()[1] << std::dec
                                            << ") from " << _tm_link;
            continue;
        }

        _lb_link.setLinkBoard(_link_record.getLinkBoard());
        _lb_link.setConnector(_link_record.getConnector());

        RPCLBLinkMap::map_type::const_iterator _lb_link_it = es_lb_link_map_->getMap().find(_lb_link);
        if (_lb_link_it == es_lb_link_map_->getMap().end()) {
            if (fill_counters_ && _bx_offset == 0) {
                _counters.add(RPCAMCLinkCounters::input_connector_not_used_, _tm_link);
            }
            LogDebug("RPCTwinMuxRawToDigi") << "Could not find " << _lb_link
                                            << " for record " << _link << " (" << std::hex << _link_record.getRecord()
                                            << " in " << _record.getRecord()[0] << ':' << _record.getRecord()[1] << std::dec
                                            << ") from " << _tm_link;
            continue;
        }

        if (_bx < _bx_min || _bx > _bx_max) {
            continue;
        }

        if (fill_counters_ && _bx == 0) {
            _counters.add(RPCAMCLinkCounters::amc_data_, RPCAMCLink(_fed, _amc_number));
            _counters.add(RPCAMCLinkCounters::input_data_, _tm_link);
        }

        RPCFebConnector const & _feb_connector(_lb_link_it->second);
        RPCDetId _det_id(_feb_connector.getRPCDetId());
        unsigned int _channel_offset(_link_record.getPartition() ? 9 : 1); // 1-16
        ::uint8_t _data(_link_record.getPartitionData());

        for (unsigned int _channel = 0 ; _channel < 8 ; ++_channel) {
            if (_data & (0x1 << _channel)) {
                unsigned int _strip(_feb_connector.getStrip(_channel + _channel_offset));
                if (_strip) {
                    _digis.insert(std::pair<RPCDetId, RPCDigi>(_det_id, RPCDigi(_strip, _bx)));
                    LogDebug("RPCTwinMuxRawToDigi") << "RPCDigi " << _det_id.rawId()
                                                    << ", " << _strip << ", " << _bx;
                }
            }
        }

        // rpctwinmux::RPCBXRecord checks postponed: not implemented in firmware as planned and tbd if design or firmware should change

    }
}

void RPCTwinMuxRawToDigi::putRPCDigis(edm::Event & _event
                                      , std::set<std::pair<RPCDetId, RPCDigi> > const & _digis)
{
    std::unique_ptr<RPCDigiCollection> _rpc_digi_collection(new RPCDigiCollection());
    RPCDetId _rpc_det_id;
    std::vector<RPCDigi> _local_digis;
    for (std::pair<RPCDetId, RPCDigi> const & _rpc_digi : _digis) {
        LogDebug("RPCTwinMuxRawToDigi") << "RPCDigi " << _rpc_digi.first.rawId()
                                        << ", " << _rpc_digi.second.strip() << ", " << _rpc_digi.second.bx();
        if (_rpc_digi.first != _rpc_det_id) {
            if (!_local_digis.empty()) {
                _rpc_digi_collection->put(RPCDigiCollection::Range(_local_digis.begin(), _local_digis.end()), _rpc_det_id);
                _local_digis.clear();
            }
            _rpc_det_id = _rpc_digi.first;
        }
        _local_digis.push_back(_rpc_digi.second);
    }
    if (!_local_digis.empty()) {
        _rpc_digi_collection->put(RPCDigiCollection::Range(_local_digis.begin(), _local_digis.end()), _rpc_det_id);
    }

    _event.put(std::move(_rpc_digi_collection));
}

void RPCTwinMuxRawToDigi::putCounters(edm::Event & _event
                                      , std::unique_ptr<RPCAMCLinkCounters>_counters)
{
    _event.put(std::move(_counters));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RPCTwinMuxRawToDigi);

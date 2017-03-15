#include "EventFilter/RPCRawToDigi/interface/RPCLBPacker.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CondFormats/RPCObjects/interface/RPCInverseLBLinkMap.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"

void RPCLBPacker::getRPCLBRecords(RPCInverseLBLinkMap const & _lb_map
                                  , int _min_bx, int _max_bx, unsigned int _bcn
                                  , RPCDigiCollection const & _digis
                                  , std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > > & _mlb_bx_lbrecord
                                  , bool _ignore_eod)
{
    if (_max_bx - _min_bx >= 128 || _min_bx < -3564) { // limit 7 bits RPCLBRecord BCN; avoid overflow
        throw cms::Exception("RPCLBPacker")
            << "Out-of-range input for _min_bx, _max_bx (" << _min_bx << ": " << _max_bx << ")";
    }

    std::map<RPCLBLink, std::vector<RPCLBRecord> > _mlb_lbrecords;

    // digi to record ; setBCN to (_bx - _min_bx) for easy sorting (avoid -1 = 3563 > 0)
    RPCDigiCollection::DigiRangeIterator _digi_range_end(_digis.end());
    for (RPCDigiCollection::DigiRangeIterator _digi_range = _digis.begin()
             ; _digi_range != _digi_range_end ; ++_digi_range) {
        RPCDigiCollection::DigiRangeIterator::value_type _digi_range_value(*_digi_range);
        std::pair<RPCInverseLBLinkMap::map_type::const_iterator
                  , RPCInverseLBLinkMap::map_type::const_iterator> _lookup_range(_lb_map.getMap().equal_range(_digi_range_value.first.rawId()));

        for (RPCDigiCollection::const_iterator _digi = _digi_range_value.second.first
                 ; _digi != _digi_range_value.second.second ; ++_digi) {
            if (_digi->bx() < _min_bx || _digi->bx() > _max_bx) {
                continue;
            }

            for (RPCInverseLBLinkMap::map_type::const_iterator _link_it = _lookup_range.first
                     ; _link_it != _lookup_range.second ; ++_link_it) {
                if (_link_it->second.second.hasStrip(_digi->strip())) {
                    unsigned int _channel(_link_it->second.second.getChannel(_digi->strip()));
                    RPCLBLink _lb_link(_link_it->second.first);
                    RPCLBLink _mlb_link(_lb_link);
                    _mlb_link.setLinkBoard().setConnector();
                    _mlb_lbrecords[_mlb_link].push_back(RPCLBRecord(_digi->bx() - _min_bx, false
                                                                    , _lb_link.getLinkBoard(), false, 0, _lb_link.getConnector()
                                                                    , (_channel > 8 ? 1 : 0), 0x01 << ((_channel - 1) % 8) ));
                    break;
                }
            }
        }
    }

    // merge records, set correct bcn and delay, and EOD if necessary
    for (std::map<RPCLBLink, std::vector<RPCLBRecord> >::iterator _mlb_lbrecords_it = _mlb_lbrecords.begin()
             ; _mlb_lbrecords_it != _mlb_lbrecords.end() ; ++_mlb_lbrecords_it) {
        std::vector<RPCLBRecord> & _input(_mlb_lbrecords_it->second);
        std::sort(_input.begin(), _input.end());

        std::vector<std::pair<int, RPCLBRecord> > _bx_lbrecord;
        _bx_lbrecord.reserve(_input.size());

        RPCLBRecord _last_lbrecord(_input.front()); // it's not empty by construction
        unsigned int _idx(0);
        for (std::vector<RPCLBRecord>::const_iterator _input_it = _input.begin() + 1
                 ; _input_it <= _input.end() ; ++_input_it) {
            if (_input_it != _input.end()
                && ((_last_lbrecord.getRecord() & ~RPCLBRecord::partition_data_mask_)
                    == (_input_it->getRecord() & ~RPCLBRecord::partition_data_mask_))) {
                _last_lbrecord.set(_last_lbrecord.getRecord() | _input_it->getRecord());
            } else {
                unsigned int _last_bcn(_last_lbrecord.getBCN());
                if (_last_bcn > _idx) {
                    _idx = _last_bcn;
                }
                unsigned int _delay(_idx - _last_bcn);
                if (_ignore_eod && _delay == 8) {
                    --_idx;
                    --_delay;
                    _bx_lbrecord.back().second.setEOD(true);
                    _last_lbrecord.setEOD(true);
                }
                if (_delay < 8) {
                    _last_lbrecord.setDelay(_delay);
                    _last_bcn = (3564 + _bcn + _min_bx + _idx) % 3564;
                    _last_lbrecord.setBCN(_last_bcn);
                    _last_lbrecord.setBC0(_last_bcn == 0);
                    _bx_lbrecord.push_back(std::pair<int, RPCLBRecord>(_min_bx + _idx, _last_lbrecord));
                    ++_idx;
                } else {
                    _bx_lbrecord.back().second.setEOD(true);
                }
                if (_input_it != _input.end()) {
                    _last_lbrecord = *_input_it;
                }
            }
        }

        _mlb_bx_lbrecord.insert(std::map<RPCLBLink
                                , std::vector<std::pair<int, RPCLBRecord> > >::value_type(_mlb_lbrecords_it->first
                                                                                          , _bx_lbrecord));
    }
}

#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxPacker.h"

#include "CondFormats/RPCObjects/interface/RPCInverseAMCLinkMap.h"
#include "EventFilter/RPCRawToDigi/interface/RPCLBPacker.h"

void RPCTwinMuxPacker::getRPCTwinMuxRecords(RPCInverseLBLinkMap const & _lb_map, RPCInverseAMCLinkMap const & _amc_map
                                            , int _min_bx, int _max_bx, unsigned int _bcn
                                            , RPCDigiCollection const & _digis
                                            , std::map<RPCAMCLink, std::vector<std::pair<int, rpctwinmux::RPCRecord> > > & _amc_bx_tmrecord
                                            , bool _ignore_eod)
{
    std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > > _mlb_bx_lbrecord;
    RPCLBPacker::getRPCLBRecords(_lb_map
                                 , _min_bx, _max_bx, _bcn
                                 , _digis
                                 , _mlb_bx_lbrecord
                                 , _ignore_eod);

    for (std::map<RPCLBLink, std::vector<std::pair<int, RPCLBRecord> > >::const_iterator _mlb_bx_lbrecord_it = _mlb_bx_lbrecord.begin()
             ; _mlb_bx_lbrecord_it != _mlb_bx_lbrecord.end() ; ++_mlb_bx_lbrecord_it) {
        // multimap, but no splitting for TwinMux inputs
        RPCInverseAMCLinkMap::map_type::const_iterator _amc_it(_amc_map.getMap().find(_mlb_bx_lbrecord_it->first));
        if (_amc_it == _amc_map.getMap().end()) {
            continue;
        }

        RPCAMCLink _amc_id(_amc_it->second);
        int _amc_input(_amc_id.getAMCInput());
        _amc_id.setAMCInput();
        std::vector<std::pair<int, rpctwinmux::RPCRecord> > & _bx_tmrecord(_amc_bx_tmrecord[_amc_id]);
        std::vector<std::pair<int, rpctwinmux::RPCRecord> >::iterator _tmrecord_it(_bx_tmrecord.begin());
        for (std::vector<std::pair<int, RPCLBRecord> >::const_iterator _bx_lbrecord = _mlb_bx_lbrecord_it->second.begin()
                 ; _bx_lbrecord != _mlb_bx_lbrecord_it->second.end() ; ++_bx_lbrecord) {
            // find the first record at this bx for this amc without this input
            for ( ; _tmrecord_it != _bx_tmrecord.end() && _tmrecord_it->first < _bx_lbrecord->first; ++_tmrecord_it);
            if (_tmrecord_it == _bx_tmrecord.end() || _tmrecord_it->first != _bx_lbrecord->first) {
                _tmrecord_it = _bx_tmrecord.insert(_tmrecord_it
                                                   , std::pair<int, rpctwinmux::RPCRecord>(_bx_lbrecord->first
                                                                                           , rpctwinmux::RPCRecord()));
                _tmrecord_it->second.setBXOffset(_bx_lbrecord->first);
            }
            rpctwinmux::RPCLinkRecord _tm_link_record;
            _tm_link_record.setAcknowledge(true);
            _tm_link_record.setEOD(_bx_lbrecord->second.isEOD());
            _tm_link_record.setDelay(_bx_lbrecord->second.getDelay());
            _tm_link_record.setLinkBoard(_bx_lbrecord->second.getLinkBoard());
            _tm_link_record.setConnector(_bx_lbrecord->second.getConnector());
            _tm_link_record.setPartition(_bx_lbrecord->second.getPartition());
            _tm_link_record.setPartitionData(_bx_lbrecord->second.getPartitionData());

            _tmrecord_it->second.setRPCLinkRecord(_amc_input, _tm_link_record);
            // make sure we don't fill this input twice if _ignore_eod == true
            ++_tmrecord_it;
        }
    }
}

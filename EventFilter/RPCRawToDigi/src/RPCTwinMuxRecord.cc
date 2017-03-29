#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

namespace rpctwinmux {

TwinMuxRecord::TwinMuxRecord(std::uint64_t const _record)
    : record_(_record)
{}

BlockHeader::BlockHeader(std::uint64_t const _record)
    : record_(_record)
{}

BlockHeader::BlockHeader(unsigned int _ufov, unsigned int _n_amc, unsigned int _orbit_counter)
    : record_(0x0)
{
    setFirmwareVersion(_ufov);
    setNAMC(_n_amc);
    setOrbitCounter(_orbit_counter);
}

BlockTrailer::BlockTrailer(std::uint64_t const _record)
    : record_(_record)
{}

BlockTrailer::BlockTrailer(std::uint32_t _crc
                           , unsigned int _block_number
                           , unsigned int _event_counter
                           , unsigned int _bx_counter)
    : record_(0x0)
{
    setCRC(_crc);
    setBlockNumber(_block_number);
    setEventCounter(_event_counter);
    setBXCounter(_bx_counter);
}

BlockAMCContent::BlockAMCContent(std::uint64_t const _record)
    : record_(_record)
{}

BlockAMCContent::BlockAMCContent(bool _length_correct
                                 , bool _last_block
                                 , bool _first_block
                                 , bool _enabled
                                 , bool _present
                                 , bool _valid
                                 , bool _crc_ok
                                 , unsigned int _size
                                 , unsigned int _block_number
                                 , unsigned int _amc_number
                                 , unsigned int _board_id)
: record_(0x0)
{
    setLengthCorrect(_length_correct);
    setLastBlock(_last_block);
    setFirstBlock(_first_block);
    setEnabled(_enabled);
    setPresent(_present);
    setValid(_valid);
    setCRCOk(_crc_ok);
    setSize(_size);
    setBlockNumber(_block_number);
    setAMCNumber(_amc_number);
    setBoardId(_board_id);
}

TwinMuxHeader::TwinMuxHeader()
{
    record_[0] = 0x0;
    record_[1] = dt_bx_window_mask_ | rpc_bx_window_mask_ | ho_bx_window_mask_;
}

TwinMuxHeader::TwinMuxHeader(std::uint64_t const _record[2])
{
    record_[0] = _record[0];
    record_[1] = _record[1];
}

TwinMuxHeader::TwinMuxHeader(unsigned int _amc_number
                             , unsigned int _event_counter
                             , unsigned int _bx_counter
                             , unsigned int _data_length
                             , unsigned int _orbit_counter
                             , unsigned int _board_id
                             , unsigned int _dt_bx_window
                             , int _rpc_bx_min, int _rpc_bx_max
                             , unsigned int _ho_bx_window)
{
    record_[0] = 0x0;
    record_[1] = dt_bx_window_mask_ | rpc_bx_window_mask_ | ho_bx_window_mask_;

    setAMCNumber(_amc_number);
    setEventCounter(_event_counter);
    setBXCounter(_bx_counter);
    setDataLength(_data_length);

    setOrbitCounter(_orbit_counter);
    setBoardId(_board_id);

    setDTBXWindow(_dt_bx_window);
    setRPCBXWindow(_rpc_bx_min, _rpc_bx_max);
    setHOBXWindow(_ho_bx_window);
}

TwinMuxTrailer::TwinMuxTrailer(std::uint64_t const _record)
    : record_(_record)
{}

TwinMuxTrailer::TwinMuxTrailer(std::uint32_t _crc
                               , unsigned int _event_counter
                               , unsigned int _data_length)
    : record_(0x0)
{
    setCRC(_crc);
    setEventCounter(_event_counter);
    setDataLength(_data_length);
}

RPCLinkRecord::RPCLinkRecord(std::uint32_t const _record)
    : record_(_record)
{}

RPCBXRecord::RPCBXRecord(std::uint8_t const _record)
    : record_(_record)
{}

unsigned int const RPCRecord::link_record_word_[] = {0, 0, 1, 1, 1};
unsigned int const RPCRecord::link_record_offset_[] = {20, 0, 40, 20, 0};
unsigned int const RPCRecord::bx_record_offset_[] = {52, 49, 46, 43, 40};

RPCRecord::RPCRecord()
{
    record_[0] = TwinMuxRecord::rpc_first_identifier_
        | ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[0])
        | ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[1]);
    record_[1] = TwinMuxRecord::rpc_second_identifier_
        | ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[2])
        | ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[3])
        | ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[4]);
}

RPCRecord::RPCRecord(std::uint64_t const _record[2])
{
    record_[0] = _record[0];
    record_[1] = _record[1];
}

} // namespace rpctwinmux

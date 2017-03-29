#include "EventFilter/RPCRawToDigi/interface/RPCLBRecord.h"

RPCLBRecord::RPCLBRecord(std::uint32_t const _record)
    : record_(_record)
{}

RPCLBRecord::RPCLBRecord(unsigned int _bcn
                         , bool _bc0
                         , unsigned int _link_board
                         , bool _eod
                         , unsigned int _delay
                         , unsigned int _connector
                         , unsigned int _partition
                         , std::uint8_t _data)
    : record_(0x00)
{
    setBCN(_bcn);
    setBC0(_bc0);
    setLinkBoard(_link_board);
    setEOD(_eod);
    setDelay(_delay);
    setConnector(_connector);
    setPartition(_partition);
    setPartitionData(_data);
}

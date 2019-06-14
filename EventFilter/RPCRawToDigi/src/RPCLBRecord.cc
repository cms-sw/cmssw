#include "EventFilter/RPCRawToDigi/interface/RPCLBRecord.h"

RPCLBRecord::RPCLBRecord(std::uint32_t const record) : record_(record) {}

RPCLBRecord::RPCLBRecord(unsigned int bcn,
                         bool bc0,
                         unsigned int link_board,
                         bool eod,
                         unsigned int delay,
                         unsigned int connector,
                         unsigned int partition,
                         std::uint8_t data)
    : record_(0x00) {
  setBCN(bcn);
  setBC0(bc0);
  setLinkBoard(link_board);
  setEOD(eod);
  setDelay(delay);
  setConnector(connector);
  setPartition(partition);
  setPartitionData(data);
}

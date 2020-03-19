#include "EventFilter/RPCRawToDigi/interface/RPCMP7Record.h"

namespace rpcmp7 {

  Header::Header() {
    record_[0] = 0x0;
    record_[1] = (std::uint64_t)(l1a_type_) << event_type_offset_;
  }

  Header::Header(std::uint64_t const record[2]) : rpcamc::Header(record) {}

  Header::Header(unsigned int amc_number,
                 unsigned int event_counter,
                 unsigned int bx_counter,
                 unsigned int data_length,
                 unsigned int orbit_counter,
                 unsigned int board_id,
                 unsigned int event_type)
      : rpcamc::Header(amc_number, event_counter, bx_counter, data_length, orbit_counter, board_id) {
    setEventType(event_type);
  }

  Header::~Header() {}

  SubHeader::SubHeader(std::uint64_t const record) : record_(record) {}

  BlockHeader::BlockHeader(std::uint32_t const record) : record_(record) {}

  BXHeader::BXHeader(std::uint32_t const record) : record_(record) {}

}  // namespace rpcmp7

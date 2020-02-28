#include "EventFilter/RPCRawToDigi/interface/RPCAMCRecord.h"

namespace rpcamc {

  Header::Header() {
    record_[0] = 0x0;
    record_[1] = 0x0;
  }

  Header::Header(std::uint64_t const record[2]) {
    record_[0] = record[0];
    record_[1] = record[1];
  }

  Header::Header(unsigned int amc_number,
                 unsigned int event_counter,
                 unsigned int bx_counter,
                 unsigned int data_length,
                 unsigned int orbit_counter,
                 unsigned int board_id) {
    record_[0] = 0x0;
    record_[1] = 0x0;

    setAMCNumber(amc_number);
    setEventCounter(event_counter);
    setBXCounter(bx_counter);
    setDataLength(data_length);

    setOrbitCounter(orbit_counter);
    setBoardId(board_id);
  }

  Header::~Header() {}

  Trailer::Trailer(std::uint64_t const record) : record_(record) {}

  Trailer::Trailer(std::uint32_t crc, unsigned int event_counter, unsigned int data_length) : record_(0x0) {
    setCRC(crc);
    setEventCounter(event_counter);
    setDataLength(data_length);
  }

}  // namespace rpcamc

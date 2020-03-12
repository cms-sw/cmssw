#include "EventFilter/RPCRawToDigi/interface/RPCAMC13Record.h"

namespace rpcamc13 {

  Header::Header(std::uint64_t const record) : record_(record) {}

  Header::Header(unsigned int ufov, unsigned int n_amc, unsigned int orbit_counter) : record_(0x0) {
    setFirmwareVersion(ufov);
    setNAMC(n_amc);
    setOrbitCounter(orbit_counter);
  }

  Trailer::Trailer(std::uint64_t const record) : record_(record) {}

  Trailer::Trailer(std::uint32_t crc, unsigned int block_number, unsigned int event_counter, unsigned int bx_counter)
      : record_(0x0) {
    setCRC(crc);
    setBlockNumber(block_number);
    setEventCounter(event_counter);
    setBXCounter(bx_counter);
  }

  AMCHeader::AMCHeader(std::uint64_t const record) : record_(record) {}

  AMCHeader::AMCHeader(bool length_correct,
                       bool last_block,
                       bool first_block,
                       bool enabled,
                       bool present,
                       bool valid,
                       bool crc_ok,
                       unsigned int size,
                       unsigned int block_number,
                       unsigned int amc_number,
                       unsigned int board_id)
      : record_(0x0) {
    setLengthCorrect(length_correct);
    setLastBlock(last_block);
    setFirstBlock(first_block);
    setEnabled(enabled);
    setPresent(present);
    setValid(valid);
    setCRCOk(crc_ok);
    setSize(size);
    setBlockNumber(block_number);
    setAMCNumber(amc_number);
    setBoardId(board_id);
  }

  AMCPayload::AMCPayload() : valid_(true) {}

}  // namespace rpcamc13

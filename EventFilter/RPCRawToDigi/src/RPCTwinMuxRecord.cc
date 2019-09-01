#include "EventFilter/RPCRawToDigi/interface/RPCTwinMuxRecord.h"

namespace rpctwinmux {

  TwinMuxRecord::TwinMuxRecord(std::uint64_t const record) : record_(record) {}

  BlockHeader::BlockHeader(std::uint64_t const record) : record_(record) {}

  BlockHeader::BlockHeader(unsigned int ufov, unsigned int n_amc, unsigned int orbit_counter) : record_(0x0) {
    setFirmwareVersion(ufov);
    setNAMC(n_amc);
    setOrbitCounter(orbit_counter);
  }

  BlockTrailer::BlockTrailer(std::uint64_t const record) : record_(record) {}

  BlockTrailer::BlockTrailer(std::uint32_t crc,
                             unsigned int block_number,
                             unsigned int event_counter,
                             unsigned int bx_counter)
      : record_(0x0) {
    setCRC(crc);
    setBlockNumber(block_number);
    setEventCounter(event_counter);
    setBXCounter(bx_counter);
  }

  BlockAMCContent::BlockAMCContent(std::uint64_t const record) : record_(record) {}

  BlockAMCContent::BlockAMCContent(bool length_correct,
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

  TwinMuxHeader::TwinMuxHeader() {
    record_[0] = 0x0;
    record_[1] = dt_bx_window_mask_ | rpc_bx_window_mask_ | ho_bx_window_mask_;
  }

  TwinMuxHeader::TwinMuxHeader(std::uint64_t const record[2]) {
    record_[0] = record[0];
    record_[1] = record[1];
  }

  TwinMuxHeader::TwinMuxHeader(unsigned int amc_number,
                               unsigned int event_counter,
                               unsigned int bx_counter,
                               unsigned int data_length,
                               unsigned int orbit_counter,
                               unsigned int board_id,
                               unsigned int dt_bx_window,
                               int rpc_bx_min,
                               int rpc_bx_max,
                               unsigned int ho_bx_window) {
    record_[0] = 0x0;
    record_[1] = dt_bx_window_mask_ | rpc_bx_window_mask_ | ho_bx_window_mask_;

    setAMCNumber(amc_number);
    setEventCounter(event_counter);
    setBXCounter(bx_counter);
    setDataLength(data_length);

    setOrbitCounter(orbit_counter);
    setBoardId(board_id);

    setDTBXWindow(dt_bx_window);
    setRPCBXWindow(rpc_bx_min, rpc_bx_max);
    setHOBXWindow(ho_bx_window);
  }

  TwinMuxTrailer::TwinMuxTrailer(std::uint64_t const record) : record_(record) {}

  TwinMuxTrailer::TwinMuxTrailer(std::uint32_t crc, unsigned int event_counter, unsigned int data_length)
      : record_(0x0) {
    setCRC(crc);
    setEventCounter(event_counter);
    setDataLength(data_length);
  }

  RPCLinkRecord::RPCLinkRecord(std::uint32_t const record) : record_(record) {}

  RPCBXRecord::RPCBXRecord(std::uint8_t const record) : record_(record) {}

  unsigned int const RPCRecord::link_record_word_[] = {0, 0, 1, 1, 1};
  unsigned int const RPCRecord::link_record_offset_[] = {20, 0, 40, 20, 0};
  unsigned int const RPCRecord::bx_record_offset_[] = {52, 49, 46, 43, 40};

  RPCRecord::RPCRecord() {
    record_[0] = TwinMuxRecord::rpc_first_identifier_ |
                 ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[0]) |
                 ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[1]);
    record_[1] = TwinMuxRecord::rpc_second_identifier_ |
                 ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[2]) |
                 ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[3]) |
                 ((std::uint64_t)RPCLinkRecord::da_mask_ << link_record_offset_[4]);
  }

  RPCRecord::RPCRecord(std::uint64_t const record[2]) {
    record_[0] = record[0];
    record_[1] = record[1];
  }

}  // namespace rpctwinmux

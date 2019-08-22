#include "EventFilter/RPCRawToDigi/interface/RPCCPPFRecord.h"

namespace rpccppf {

  RXRecord::RXRecord(std::uint32_t const record) : record_(record) {}

  std::uint32_t const TXRecord::theta_mask_[] = {0x0000f800, 0xf8000000};
  std::uint32_t const TXRecord::phi_mask_[] = {0x000007ff, 0x07ff0000};
  unsigned int const TXRecord::theta_offset_[] = {11, 27};
  unsigned int const TXRecord::phi_offset_[] = {0, 16};

  TXRecord::TXRecord() : record_(phi_mask_[0] | phi_mask_[1]) {}

  TXRecord::TXRecord(std::uint32_t const record) : record_(record) {}

}  // namespace rpccppf

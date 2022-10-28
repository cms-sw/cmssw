
#include "L1Trigger/DemonstratorTools/interface/Frame.h"

namespace l1t::demo {

  Frame::Frame(const uint64_t x) : data(x), valid(true) {}

  Frame::Frame(const ap_uint<64>& x) : data(x), valid(true) {}

}  // namespace l1t::demo
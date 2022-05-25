#ifndef L1Trigger_DemonstratorTools_ChannelSpec_h
#define L1Trigger_DemonstratorTools_ChannelSpec_h

#include <cstddef>

namespace l1t::demo {

  struct ChannelSpec {
  public:
    // Time multiplexing period of data on link
    size_t tmux;
    // Number of invalid frames between packets (i.e. following each event)
    size_t interpacketGap;
    // Number of invalid frames before first valid packet
    size_t offset{0};
  };

}  // namespace l1t::demo

#endif
#ifndef L1Trigger_DemonstratorTools_LinkId_h
#define L1Trigger_DemonstratorTools_LinkId_h

#include <cstddef>
#include <string>

namespace l1t::demo {

  //! Logical ID for link within any given time slice (e.g. ["tracks", 0] -> ["tracks", 17] for links from TF)
  struct LinkId {
    std::string interface;
    size_t channel{0};
  };

  bool operator<(const LinkId&, const LinkId&);

}  // namespace l1t::demo

#endif
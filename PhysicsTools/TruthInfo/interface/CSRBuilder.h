// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_CSRBuilder_h
#define PhysicsTools_TruthInfo_interface_CSRBuilder_h

#include <cstdint>
#include <utility>
#include <vector>

namespace truth {

  // Fills one CSR adjacency of truth::Graph from (source, target) pairs in any order.
  // offsets gets nSources + 1 entries, and the targets of each source are ascending and
  // unique. A pair whose source is not below nSources is dropped.
  void buildCSR(uint32_t nSources,
                std::vector<std::pair<uint32_t, uint32_t>> const& pairs,
                std::vector<uint32_t>& offsets,
                std::vector<uint32_t>& flat);

}  // namespace truth

#endif

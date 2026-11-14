// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#include "PhysicsTools/TruthInfo/interface/CSRBuilder.h"

#include <algorithm>

namespace truth {

  void buildCSR(uint32_t nSources,
                std::vector<std::pair<uint32_t, uint32_t>> const& pairs,
                std::vector<uint32_t>& offsets,
                std::vector<uint32_t>& flat) {
    offsets.assign(nSources + 1, 0);
    for (auto const& [source, target] : pairs) {
      if (source < nSources)
        ++offsets[source + 1];
    }
    for (uint32_t i = 1; i <= nSources; ++i)
      offsets[i] += offsets[i - 1];

    flat.resize(offsets[nSources]);
    std::vector<uint32_t> cursor(offsets.begin(), offsets.end() - 1);
    for (auto const& [source, target] : pairs) {
      if (source < nSources)
        flat[cursor[source]++] = target;
    }

    // Sort and deduplicate each row, and close the gaps that the duplicates leave.
    uint32_t write = 0;
    for (uint32_t source = 0; source < nSources; ++source) {
      const auto first = flat.begin() + offsets[source];
      auto last = flat.begin() + offsets[source + 1];
      std::sort(first, last);
      last = std::unique(first, last);
      const auto size = static_cast<uint32_t>(last - first);
      if (write != offsets[source])
        std::move(first, last, flat.begin() + write);
      offsets[source] = write;
      write += size;
    }
    offsets[nSources] = write;
    flat.resize(write);
  }

}  // namespace truth

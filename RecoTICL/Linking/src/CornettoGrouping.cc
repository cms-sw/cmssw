#include "RecoTICL/Linking/interface/CornettoGrouping.h"

std::vector<unsigned int> ticl::cornetto::roots(const std::vector<unsigned int>& parents) {
  const auto n = static_cast<unsigned int>(parents.size());
  std::vector<unsigned int> root(parents);
  std::vector<unsigned int> next(n);
  for (bool changed = true; changed;) {
    changed = false;
    for (unsigned int i = 0; i < n; ++i) {
      next[i] = root[root[i]];
      changed |= next[i] != root[i];
    }
    root.swap(next);
  }
  return root;
}

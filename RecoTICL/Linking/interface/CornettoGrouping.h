#ifndef RecoTICL_Linking_CornettoGrouping_h
#define RecoTICL_Linking_CornettoGrouping_h

// Grouping of the Cornetto parent tree: the rank of a trackster, the roots by pointer jumping, and the groups. Each
// step is independent per trackster.

#include <vector>

namespace ticl::cornetto {

  // a ranks above b when it has more energy, or the same energy and a smaller index.
  inline bool ranksAbove(const std::vector<float>& energy, unsigned int a, unsigned int b) {
    return energy[a] > energy[b] || (energy[a] == energy[b] && a < b);
  }

  // The root of each node by pointer jumping on the parents; a root is its own parent. Each round reads the result of
  // the previous round only. The parents must form a forest: every chain of parents ends at a root.
  std::vector<unsigned int> roots(const std::vector<unsigned int>& parents);

  // The groups: a node stays with its root when keep(root, node) holds, else it is a group of its own. The groups are
  // in the order of their smallest member and hold their members in ascending order. Each node is in one group.
  template <typename Keep>
  std::vector<std::vector<unsigned int>> groups(const std::vector<unsigned int>& root, const Keep& keep) {
    const auto n = static_cast<unsigned int>(root.size());
    std::vector<int> groupOfLabel(n, -1);
    std::vector<std::vector<unsigned int>> out;
    for (unsigned int i = 0; i < n; ++i) {
      const unsigned int label = (root[i] == i || keep(root[i], i)) ? root[i] : i;
      int& g = groupOfLabel[label];
      if (g < 0) {
        g = static_cast<int>(out.size());
        out.emplace_back();
      }
      out[g].push_back(i);
    }
    return out;
  }

}  // namespace ticl::cornetto

#endif

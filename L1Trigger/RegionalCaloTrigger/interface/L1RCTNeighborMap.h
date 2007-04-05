#ifndef L1RCTNeighborMap_h
#define L1RCTNeighborMap_h
#include <vector>

class L1RCTNeighborMap {
 public:
  std::vector<int> north(int crate, int card, int region);
  std::vector<int> south(int crate, int card, int region);
  std::vector<int> west(int crate, int card, int region);
  std::vector<int> east(int crate, int card, int region);
  std::vector<int> se(int crate, int card, int region);
  std::vector<int> sw(int crate, int card, int region);
  std::vector<int> ne(int crate, int card, int region);
  std::vector<int> nw(int crate, int card, int region);
};
#endif

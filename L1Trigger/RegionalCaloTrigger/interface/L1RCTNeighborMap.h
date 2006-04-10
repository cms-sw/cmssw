#ifndef L1RCTNeighborMap_h
#define L1RCTNeighborMap_h
#include <vector>
using std::vector;

class L1RCTNeighborMap {
 public:
  vector<int> north(int crate, int card, int region);
  vector<int> south(int crate, int card, int region);
  vector<int> west(int crate, int card, int region);
  vector<int> east(int crate, int card, int region);
  vector<int> se(int crate, int card, int region);
  vector<int> sw(int crate, int card, int region);
  vector<int> ne(int crate, int card, int region);
  vector<int> nw(int crate, int card, int region);
};
#endif

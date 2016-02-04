#include <map>

#include "DataFormats/Common/interface/SortedCollection.h"


// A very simple (but not built-in KEY type)
struct Key
{
  int i;
};


// KEYS must be less-than comparable.
// This means, if k1 and k2 are objects of type K,
// then the expression
//     k1 < k2
// must define a strict weak ordering, and the result
// must be convertible to bool.
// Numeric types automatically satisfy this.

bool operator< (Key const& a, Key const& b) { return a.i < b.i; }

struct V
{
  typedef Key key_type;

  int i;
  key_type k;

  key_type id() const { return k; }  
};

struct BackwardsOrdering
{
  typedef Key key_type;
  bool operator()(key_type a, V const& b) const { return b.id() < a; }
  bool operator()(V const& a, key_type b) const { return b < a.id(); }
  bool operator()(V const& a, V const& b) const { return b.id() < a.id(); }  
};

int main()
{
  edm::SortedCollection<V> sc;
  std::map<Key,V>          m;

  // Fill with 2000 entries; includes sorting.

  // Do 2000 finds

  return 0;
}

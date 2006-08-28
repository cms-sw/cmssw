#include <cassert>
#include "FWCore/Utilities/interface/value_ptr.h"

class simple
{
  int i;
 public:
  static int count;
  simple() : i(0) { ++count; }
  explicit simple(int j) : i(j) { ++count; }
  simple(simple const& s) : i(s.i) { ++count; }
  ~simple() { --count; }
  bool operator==(simple const& o) const { return i == o.i; }
  bool isSame(simple const& o) const {return &o == this; }
};

int simple::count = 0;


int main()
{
  assert(simple::count == 0);
  {
    edm::value_ptr<simple> a(new simple(10));
    assert(simple::count == 1);
    edm::value_ptr<simple> b(a);
    assert(simple::count == 2);
    
    assert(*a==*b);
    assert(a->isSame(*b) == false);
  } // a and b destroyed
  assert(simple::count == 0);
}

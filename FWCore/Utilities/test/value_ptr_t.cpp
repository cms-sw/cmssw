#include "FWCore/Utilities/interface/value_ptr.h"

#include <memory>
#include <cassert>

class simple
{
  int i;
 public:
  static int count;
  static int count1;
  simple() : i(0) { ++count; ++count1; }
  explicit simple(int j) : i(j) { ++count; ++count1; }
  simple(simple const& s) : i(s.i) { ++count; ++count1; }
  ~simple() { --count; }
  bool operator==(simple const& o) const { return i == o.i; }
  bool isSame(simple const& o) const {return &o == this; }
};

int simple::count = 0;
int simple::count1 = 0;


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

  {
    std::auto_ptr<simple> c(new simple(11));
    std::auto_ptr<simple> d(new simple(11));
    assert(c.get() != 0);
    assert(d.get() != 0);
    simple* pc = c.get();
    simple* pd = d.get();

    edm::value_ptr<simple> e(c);
    assert(c.get() == 0);
    assert(*d == *e);
    assert(e.operator->() == pc);

    edm::value_ptr<simple> f;
    if (f) {
      assert(0);
    }
    else {
    }
    f = d;
    assert(d.get() == 0);
    assert(*e == *f);
    assert(f.operator->() == pd);
    if (f) {
    }
    else {
      assert(0);
    }

    assert(simple::count == 2);
    assert(simple::count1 == 4);
  }
  assert(simple::count == 0);
  assert(simple::count1 == 4);
}

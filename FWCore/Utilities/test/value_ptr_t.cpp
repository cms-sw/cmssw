#include "FWCore/Utilities/interface/value_ptr.h"

#include <memory>
#include <cassert>
#include <atomic>

class simple
{
  int i;
 public:
  static std::atomic<int> count;
  static std::atomic<int> count1;
  simple() : i(0) { ++count; ++count1; }
  explicit simple(int j) : i(j) { ++count; ++count1; }
  simple(simple const& s) : i(s.i) { ++count; ++count1; }
  ~simple() { --count; }
  bool operator==(simple const& o) const { return i == o.i; }
  bool isSame(simple const& o) const {return &o == this; }
  simple& operator=(simple const& o) {
    simple temp(o);
    std::swap(*this, temp);
    return *this;
  }
};

std::atomic<int> simple::count{0};
std::atomic<int> simple::count1{0};


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
    assert(c.get() != nullptr);
    assert(d.get() != nullptr);
    simple* pc = c.get();
    simple* pd = d.get();

    edm::value_ptr<simple> e(c);
    assert(c.get() == nullptr);
    assert(*d == *e);
    assert(e.operator->() == pc);

    edm::value_ptr<simple> f;
    if (f) {
      assert(0);
    }
    else {
    }
    f = d;
    assert(d.get() == nullptr);
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

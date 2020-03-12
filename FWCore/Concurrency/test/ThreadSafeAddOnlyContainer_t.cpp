#include "FWCore/Concurrency/interface/ThreadSafeAddOnlyContainer.h"

#include <iostream>
#include <string>

namespace {
  template <typename U, typename V>
  class X {
  public:
    X(U const& a, V const& b, double c) : a_(a), b_(b), c_(c) { std::cout << "Constructing " << a_ << std::endl; }
    ~X() { std::cout << "~X " << a_ << std::endl; }
    U a_;
    V b_;
    double c_;
  };

  class Y {
  public:
    Y() { std::cout << "constructing Y" << std::endl; }
    ~Y() { std::cout << "~Y" << std::endl; }
  };
}  // namespace

int main() {
  edm::ThreadSafeAddOnlyContainer<int> container1;
  int* ptr1 = container1.makeAndHold(11);
  // std::cout << *ptr1 << std::endl;
  if (*ptr1 != 11)
    abort();

  edm::ThreadSafeAddOnlyContainer<X<std::string, int> > container2;
  X<std::string, int>* ptr2 = container2.makeAndHold(std::string("FOO"), 11, 21.0);
  // std::cout << ptr2->a_ << " " << ptr2->b_ << " " << ptr2->c_ << std::endl;
  if (ptr2->a_ != "FOO" || ptr2->b_ != 11 || ptr2->c_ != 21.0)
    abort();

  X<std::string, int>* ptr3 = container2.makeAndHold(std::string("BAR"), 111, 121.0);
  // std::cout << ptr3->a_ << " " << ptr3->b_ << " " << ptr3->c_ << std::endl;
  if (ptr3->a_ != "BAR" || ptr3->b_ != 111 || ptr3->c_ != 121.0)
    abort();

  edm::ThreadSafeAddOnlyContainer<X<std::string, int> > container3;

  edm::ThreadSafeAddOnlyContainer<Y> container4;
  container4.makeAndHold();
  container4.makeAndHold();

  return 0;
}

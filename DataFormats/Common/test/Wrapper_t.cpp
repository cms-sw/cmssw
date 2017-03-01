/*
 *  CMSSW
 *
 */

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"

class CopyNoSwappy
{
 public:
  CopyNoSwappy() {}
  CopyNoSwappy(CopyNoSwappy const&) { /* std::cout << "copied\n"; */ }
  CopyNoSwappy& operator=(CopyNoSwappy const&) { /*std::cout << "assigned\n";*/ return *this;}
 private:
};

class SwappyNoCopy
{
 public:
  SwappyNoCopy() {}
  void swap(SwappyNoCopy&) { /* std::cout << "swapped\n";*/ }
 private:
  SwappyNoCopy(SwappyNoCopy const&); // not implemented
  SwappyNoCopy& operator=(SwappyNoCopy const&); // not implemented
};

void work()
{
  auto thing = std::make_unique<CopyNoSwappy>();
  edm::Wrapper<CopyNoSwappy> wrap(std::move(thing));

  auto thing2 = std::make_unique<SwappyNoCopy>();
  edm::Wrapper<SwappyNoCopy> wrap2(std::move(thing2));


  auto thing3 = std::make_unique<std::vector<double>>(10,2.2);
  assert(thing3->size() == 10);

  edm::Wrapper<std::vector<double> > wrap3(std::move(thing3));
  assert(wrap3->size() == 10);
  assert(thing3.get() == 0);
}

int main()
{
  int rc = 0;
  try {
      work();
  }
  catch (...) {
      rc = 1;
      std::cerr << "Failure: unidentified exception caught\n";
  }
  return rc;
}

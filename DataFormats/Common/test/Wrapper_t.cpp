/*
 *  $Id: Wrapper_t.cpp,v 1.1 2006/08/07 23:44:02 wmtan Exp $
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
  CopyNoSwappy(CopyNoSwappy const& ) { std::cout << "copied\n"; }
  CopyNoSwappy& operator=(CopyNoSwappy const& ) {std::cout << "assigned\n"; return *this;}
 private:
#if ! __GNUC_PREREQ (3,4)
  void swap(CopyNoSwappy&); // not implemented
#endif
};

class SwappyNoCopy
{
 public:
  SwappyNoCopy() {}
  void swap(SwappyNoCopy&) { std::cout << "swapped\n"; }
 private:
  SwappyNoCopy(SwappyNoCopy const&); // not implemented
  SwappyNoCopy& operator=(SwappyNoCopy const&); // not implemented
};

#if ! __GNUC_PREREQ (3,4)
namespace edm
{
  template <> struct has_swap<SwappyNoCopy> { static const bool value=true; };
}
#endif

void work()
{
  std::auto_ptr<CopyNoSwappy> thing(new CopyNoSwappy);
  edm::Wrapper<CopyNoSwappy> wrap(thing);

  std::auto_ptr<SwappyNoCopy> thing2(new SwappyNoCopy);
  edm::Wrapper<SwappyNoCopy> wrap2(thing2);


  std::auto_ptr<std::vector<double> > 
    thing3(new std::vector<double>(10,2.2));
  assert(thing3->size() == 10);

  edm::Wrapper<std::vector<double> > wrap3(thing3);
  assert(wrap3->size() == 10);
  assert(thing3.get() == 0);
}

int main()
{
  int rc = 0;
  try {
      work();
  }
  catch ( ... ) {
      rc = 1;
      std::cerr << "Failure: unidentified exception caught\n";
  }
  return rc;
}

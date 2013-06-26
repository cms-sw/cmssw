
#include "DataFormats/TestObjects/interface/StreamTestThing.h"

#include <algorithm>
#include <cstdlib>

namespace {
  struct Setter {
    ~Setter() { }
    Setter() { srand(1011); }
  };
}

namespace edmtestprod {

 StreamTestThing::~StreamTestThing() { }

 StreamTestThing::StreamTestThing() : data_()
 {
 }

 StreamTestThing::StreamTestThing(int sz) : data_(sz)
 {
    static Setter junker;
    generate(data_.begin(),data_.end(),rand);
 }

}

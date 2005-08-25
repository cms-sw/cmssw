
#include "IOPool/Streamer/interface/StreamTestThing.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>

using namespace std;

namespace {
  struct Setter {
    ~Setter() { }
    Setter() { srand(1011); }
  };
}

namespace edmtestprod {

 StreamTestThing::~StreamTestThing() { }

 StreamTestThing::StreamTestThing()
 {
 }

 StreamTestThing::StreamTestThing(int sz):
	data_(sz)
 {
    static Setter junker;
    generate(data_.begin(),data_.end(),rand);
 }

}

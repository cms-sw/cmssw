#ifndef Streamer_StreamTestThing_h
#define Streamer_StreamTestThing_h

#include <vector>

namespace edmtestprod {

  struct StreamTestThing
  {
    ~StreamTestThing();
    explicit StreamTestThing(int sz);
    StreamTestThing();

    std::vector<int> data_;
  };

}

#endif

#ifndef TestObjects_StreamTestThing_h
#define TestObjects_StreamTestThing_h

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

#ifndef EDMREFTEST_THING_H
#define EDMREFTEST_THING_H

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

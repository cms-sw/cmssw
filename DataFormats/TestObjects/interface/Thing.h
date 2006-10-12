#ifndef TestObjects_Thing_h
#define TestObjects_Thing_h
#include <vector>

namespace edmtest {

  struct Thing {
    ~Thing() { }
    Thing():a() { }
    int a;
  };

}

#endif

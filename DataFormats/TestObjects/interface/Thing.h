#ifndef TestObjects_Thing_h
#define TestObjects_Thing_h
//INCLUDECHECKER: Removed this line: #include <vector>

namespace edmtest {

  struct Thing {
    ~Thing() { }
    Thing():a() { }
    int a;
  };

}

#endif

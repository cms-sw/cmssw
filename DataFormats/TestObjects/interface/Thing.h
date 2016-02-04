#ifndef TestObjects_Thing_h
#define TestObjects_Thing_h

namespace edmtest {

  struct Thing {
    ~Thing() { }
    Thing():a() { }
    int a;
  };

}

#endif

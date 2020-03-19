#ifndef DataFormats_TestObjects_MissingDictionaryTestObject_h
#define DataFormats_TestObjects_MissingDictionaryTestObject_h

#include <vector>

namespace edmtest {

  class MissingDictionaryTestK {
  public:
    ~MissingDictionaryTestK() {}
    MissingDictionaryTestK() : k(0) {}
    int k;
  };

  class MissingDictionaryTestJ {
  public:
    ~MissingDictionaryTestJ() {}
    MissingDictionaryTestJ() : j(0) {}
    int j;
  };

  class MissingDictionaryTestI : public MissingDictionaryTestJ {
  public:
    ~MissingDictionaryTestI() {}
    MissingDictionaryTestI() : i(0) {}
    int i;
    MissingDictionaryTestK k;
  };

  class MissingDictionaryTestH {
  public:
    ~MissingDictionaryTestH() {}
    MissingDictionaryTestH() : h(0) {}
    int h;
  };

  class MissingDictionaryTestG {
  public:
    ~MissingDictionaryTestG() {}
    MissingDictionaryTestG() : g(0) {}
    int g;
  };

  class MissingDictionaryTestF : public MissingDictionaryTestG {
  public:
    ~MissingDictionaryTestF() {}
    MissingDictionaryTestF() : f(0) {}
    int f;
    MissingDictionaryTestH h;
  };

  class MissingDictionaryTestE {
  public:
    ~MissingDictionaryTestE() {}
    MissingDictionaryTestE() : e(0) {}
    int e;
  };

  class MissingDictionaryTestD {
  public:
    ~MissingDictionaryTestD() {}
    MissingDictionaryTestD() : d(0) {}
    int d;
  };

  class MissingDictionaryTestC {
  public:
    ~MissingDictionaryTestC() {}
    MissingDictionaryTestC() : c(0) {}
    int c;
    MissingDictionaryTestD d;
  };

  class MissingDictionaryTestB : public MissingDictionaryTestC {
  public:
    ~MissingDictionaryTestB() {}
    MissingDictionaryTestB() : b(0) {}
    int b;
    MissingDictionaryTestE e;
  };

  class MissingDictionaryTestA : public MissingDictionaryTestB {
  public:
    ~MissingDictionaryTestA() {}
    MissingDictionaryTestA() : a(0) {}
    int a;
    MissingDictionaryTestF f;
    std::vector<MissingDictionaryTestI> vi;
  };

}  // namespace edmtest

#endif

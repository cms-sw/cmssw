#ifndef HeterogeneousCore_AlpakaTest_interface_ESTestData_h
#define HeterogeneousCore_AlpakaTest_interface_ESTestData_h

namespace cms::alpakatest {
  class ESTestDataA {
  public:
    ESTestDataA(int v) { value_ = v; }
    int value() const { return value_; }

  private:
    int value_;
  };

  class ESTestDataB {
  public:
    ESTestDataB(int v) { value_ = v; }
    int value() const { return value_; }

  private:
    int value_;
  };

  class ESTestDataC {
  public:
    ESTestDataC(int v) { value_ = v; }
    int value() const { return value_; }

  private:
    int value_;
  };
}  // namespace cms::alpakatest

#endif

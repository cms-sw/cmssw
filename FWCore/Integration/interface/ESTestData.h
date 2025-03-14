#ifndef FWCore_Integration_ESTestData_h
#define FWCore_Integration_ESTestData_h

namespace edmtest {

  class ESTestDataA {
  public:
    ESTestDataA(int v) { value_ = v; }
    int& value() { return value_; }
    int const& value() const { return value_; }

  private:
    int value_;
  };

  class ESTestDataB {
  public:
    ESTestDataB(int v = 0, int w = 0) { value_ = v + w; }
    int& value() { return value_; }
    int const& value() const { return value_; }

  private:
    int value_;
  };

  class ESTestDataI {
  public:
    ESTestDataI(int v) { value_ = v; }
    int& value() { return value_; }
    int const& value() const { return value_; }

  private:
    int value_;
  };

  class ESTestDataJ {
  public:
    ESTestDataJ(int v) { value_ = v; }
    int& value() { return value_; }
    int const& value() const { return value_; }

  private:
    int value_;
  };

}  // namespace edmtest
#endif

#ifndef FWCore_Reflection_test_TestObjects_h
#define FWCore_Reflection_test_TestObjects_h

namespace edmtest::reflection {
  class IntObject {
  public:
    IntObject();
    IntObject(int v) : value_(v) {}

    int get() const { return value_; }

  private:
    int value_ = 0;
  };
}  // namespace edmtest::reflection

#endif

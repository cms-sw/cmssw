#ifndef FWCore_Reflection_test_TestObjects_h
#define FWCore_Reflection_test_TestObjects_h

namespace edmtest::reflection {
  class IntObject {
  public:
    IntObject();
    IntObject(int v) : value_(v) {}

#ifdef FWCORE_REFLECTION_TEST_INTOBJECT_V4
    void set(int v) {
      value_ = v;
      set_ = true;
    }
#endif
    int get() const { return value_; }

  private:
    int value_ = 0;
#ifdef FWCORE_REFLECTION_TEST_INTOBJECT_V4
    bool set_ = false;
#endif
  };
}  // namespace edmtest::reflection

#endif

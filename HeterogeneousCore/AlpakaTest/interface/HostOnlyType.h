#ifndef HeterogeneousCore_AlpakaTest_interface_HostOnlyType_h
#define HeterogeneousCore_AlpakaTest_interface_HostOnlyType_h

namespace alpakatest {

  /* A simple class to demonstarte the dependency on host-only types from alpaka libraries */
  class HostOnlyType {
  public:
    HostOnlyType() : value_{0} {}
    HostOnlyType(int value) : value_{value} {}
    void set(int value) { value_ = value; }
    int get() { return value_; }
    void print();

  private:
    int value_;
  };

}  // namespace alpakatest

#endif  // HeterogeneousCore_AlpakaTest_interface_HostOnlyType_h

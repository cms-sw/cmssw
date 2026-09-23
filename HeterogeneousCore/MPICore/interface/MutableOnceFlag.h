#ifndef HeterogeneousCore_MPICore_interface_MutableOnceFlag_h
#define HeterogeneousCore_MPICore_interface_MutableOnceFlag_h
#include <mutex>

struct MutableOnceFlag {
  //Using mutable since we want to update the value.
  mutable std::once_flag information_recorded_flag;
};

#endif  // HeterogeneousCore_MPICore_interface_MutableOnceFlag_h

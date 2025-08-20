#ifndef HeterogeneousCore_CUDATest_interface_MissingDictionaryCUDAObject_h
#define HeterogeneousCore_CUDATest_interface_MissingDictionaryCUDAObject_h

#include <string>

namespace edmtest {

  // A simple data product used to test that the framework handles correctly the case of
  // edm::Wrapper<T> where
  //   - T has a dictionary
  //   - edm::Wrapper<T> does not have a dictionary
  //   - the corresponding classes.h file includes CUDA headers

  struct MissingDictionaryCUDAObject {
    MissingDictionaryCUDAObject() {};
    MissingDictionaryCUDAObject(std::string s) : value(std::move(s)) {}

    std::string value;
  };

}  // namespace edmtest

#endif  // HeterogeneousCore_CUDATest_interface_MissingDictionaryCUDAObject_h

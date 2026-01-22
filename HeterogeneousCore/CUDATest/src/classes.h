/*
A simple data product used to test that the framework handles correctly the case of
edm::Wrapper<T> where
  - T has a dictionary
  - edm::Wrapper<T> does not have a dictionary
  - the corresponding classes.h file includes CUDA headers
*/
#include <cuda_runtime.h>

#include "DataFormats/Common/interface/Wrapper.h"
#include "HeterogeneousCore/CUDATest/interface/MissingDictionaryCUDAObject.h"

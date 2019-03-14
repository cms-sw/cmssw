#include <cstdlib>

#include "HeterogeneousCore/CUDAUtilities/interface/supportedCUDADevices.h"

int main() {
  return supportedCUDADevices().empty() ? EXIT_FAILURE : EXIT_SUCCESS;
}

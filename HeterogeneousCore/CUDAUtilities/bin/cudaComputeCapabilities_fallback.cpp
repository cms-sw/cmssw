// C/C++ headers
#include <cstdlib>
#include <iostream>

// always returns EXIT_FAILURE
int main() {
  std::cerr << "cudaComputeCapabilities: CUDA is not supported on this architecture" << std::endl;
  return EXIT_FAILURE;
}

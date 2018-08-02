#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"


#include "cuda/api_wrappers.h"
#include<cassert>

__global__
void testBinaryFind()
{
    int data[] = { 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6 };
 
    auto lower = cuda_std::lower_bound(data, data+13, 4);
    auto upper = cuda_std::upper_bound(data, data+12, 4);
 
    assert(3 == upper-lower);
 
    // classic binary search, returning a value only if it is present
 
    constexpr int data2[] = { 1, 2, 4, 6, 9, 10 }; 
 
    assert(data2+2 == cuda_std::binary_find(data2, data2+6, 4)); 
    assert(data2+6 == cuda_std::binary_find(data2, data2+6, 5));
}

#include<iostream>
void wrapper()
{

  if (cuda::device::count() == 0) {
        std::cerr << "No CUDA devices on this system" << "\n";
        exit(EXIT_FAILURE);
  }

  auto current_device = cuda::device::current::get();

  cuda::launch(
                testBinaryFind,
                { 32, 64 }
              );


}

int main() {

  wrapper();


}

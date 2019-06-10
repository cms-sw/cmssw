#include "HeterogeneousCore/CUDAUtilities/interface/cudastdAlgorithm.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
 
void testBinaryFind()
{
    std::vector<int> data = { 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6 };
 
    auto lower = cuda_std::lower_bound(data.begin(), data.end(), 4);
    auto upper = cuda_std::upper_bound(data.begin(), data.end(), 4);
 
    std::copy(lower, upper, std::ostream_iterator<int>(std::cout, " "));
 
    std::cout << '\n';
 
    // classic binary search, returning a value only if it is present
 
    data = { 1, 2, 4, 6, 9, 10 }; 
 
    auto test = [&](auto v) {
      auto it = cuda_std::binary_find(data.cbegin(), data.cend(), v); 
 
      if(it != data.cend())
        std::cout << *it << " found at index "<< std::distance(data.cbegin(), it) << std::endl;
      else 
        std::cout << v << " non found" << std::endl;
    };

    test(4);
    test(5);

}

int main() {

  testBinaryFind();


}

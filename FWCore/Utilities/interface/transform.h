#ifndef FWCore_Utilities_transform_h
#define FWCore_Utilities_transform_h

#include <vector>

namespace edm {

  // helper template function to build a vector applying a transformation to the elements of an input vector
  template <typename InputType, typename Function>
  auto vector_transform(std::vector<InputType> const & input, Function predicate) -> std::vector<decltype(predicate(input.front()))>
  {
    using ReturnType = decltype(predicate(input.front()));
    std::vector<ReturnType> output;
    output.reserve( input.size() );
    for (auto const & element : input)
      output.push_back(predicate(element));
    return output;
  }

} // namespace edm

#endif // FWCore_Utilities_transform_h

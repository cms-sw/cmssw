#ifndef FWCore_Utilities_transform_h
#define FWCore_Utilities_transform_h

#include <vector>
#include <type_traits>

namespace edm {

  // helper template function to build a vector applying a transformation to the elements of an input vector
  template <typename InputType, typename Function>
  auto vector_transform(std::vector<InputType> const & input, Function predicate) -> std::vector<typename std::remove_cv<typename std::remove_reference<decltype(predicate(input.front()))>::type>::type>
  {
    using ReturnType = typename std::remove_cv<typename std::remove_reference<decltype(predicate(input.front()))>::type>::type;
    std::vector<ReturnType> output;
    output.reserve( input.size() );
    for (auto const & element : input)
      output.push_back(predicate(element));
    return output;
  }

} // namespace edm

#endif // FWCore_Utilities_transform_h

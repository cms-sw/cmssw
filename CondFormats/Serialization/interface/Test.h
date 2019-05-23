#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include <stdexcept>

#include "CondFormats/Serialization/interface/Archive.h"
#include "CondFormats/Serialization/interface/Equal.h"

// The compiler knows our default-constructed objects' members
// may not be initialized when we serialize them.
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"

// The main test: constructs an object using the default constructor,
// serializes it, deserializes it and finally checks whether they are equal.
// Mainly used to see if all the required templates compile.
//
// It is not possible to use (easily) the Boost text archive because:
//
//   * Uninitialized booleans are serialized as a byte (unsigned char),
//     while on deserialization Boost asserts on the value being 0 or 1.
//
//   * Uninitialized floating point variables may be NaN (or Inf),
//     which are serialized as "nan" and Boost fails to deserialize them.
//
// While the problems could be solved adding default constructors
// or modifying them (C++11's in-class member initializers cannot be used
// due to genreflex in ROOT 5 which would have been a bit easier),
// for the floating point case there are too many members and classes
// to modify. Instead, the Boost binary archive can be used, which does not
// assert. However, this means it is required to support comparing NaNs and
// Infs properly in cond::serialization::equal.
//
// The EOS' portable binary archive has the same problem with uninitialized
// booleans as the Boost text archive, but not the uninitialized floating
// point variables. Since the number of boolean members in CondFormats
// is small, we initialized them in the default constructor or disabled
// the test in other cases.
//
// Note that classes with STL containers of other classes may not be tested
// (at runtime, since they have size 0 by default), unless they are explicitly
// tested by themselves (which should be the case, since in the XML it was
// required to write the "dependencies").
template <typename T>
void testSerialization() {
  const std::string filename(std::string(typeid(T).name()) + ".bin");

  // C++ does not allow to construct const objects
  // of non-POD types without user-provided default constructor
  // (since it would be uninitialized), so we always create
  // a non-const object.
  T originalObject{};
  const T& originalObjectRef = originalObject;
  {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    cond::serialization::OutputArchive oa(ofs);
    std::cout << "Serializing " << typeid(T).name() << " ..." << std::endl;
    oa << originalObjectRef;
  }

  T deserializedObject;
  {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    cond::serialization::InputArchive ia(ifs);
    std::cout << "Deserializing " << typeid(T).name() << " ..." << std::endl;
    ia >> deserializedObject;
  }

  // TODO: First m,ake the Boost IO compile and run properly,
  //       then focus again on the equal() functions.
  //std::cout << "Checking " << typeid(T).name() << " ..." << std::endl;
  //if (not cond::serialization::equal(originalObject, deserializedObject))
  //    throw std::logic_error("Object is not equal.");
}

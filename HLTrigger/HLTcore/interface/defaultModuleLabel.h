#ifndef defaultModuleLabel_h
#define defaultModuleLabel_h

// C++ headers
#include <string>

// boost headers
#include <boost/version.hpp>
#if BOOST_VERSION < 105200
#include <boost/units/detail/utility.hpp>
#else
#include <boost/core/demangle.hpp>
#endif

template <typename T>
std::string defaultModuleLabel() {
  // start with the demangled name for T
#if BOOST_VERSION < 105200
  std::string name = boost::units::detail::demangle(typeid(T).name());
#else
  std::string name = boost::core::demangle(typeid(T).name());
#endif

  // expected size of the label
  unsigned int size = 0;
  for (char c: name)
    if (std::isalnum(c)) ++size;
  std::string label;
  label.reserve(size);

  // tokenize the demangled name, keeping only alphanumeric characters,
  // and convert the tokens to lowerCamelCase.
  bool new_token = false;
  for (char c: name) {
    if (std::isalnum(c)) {
      if (new_token)
        label.push_back((char) std::toupper(c));
      else
        label.push_back(c);
      new_token = false;
    }
    else {
      new_token = true;
    }
  }

  // if the label is all uppercase, change it to all lowercase
  // if the label starts with more than one uppercase letter, change n-1 to lowercase
  // otherwise, change the first letter to lowercase
  unsigned int ups = 0;
  for (char c: label)
    if (std::isupper(c))
      ++ups;
    else
      break;
  if (ups > 1 and ups != label.size())
    --ups;
  for (unsigned int i = 0; i < ups; ++i)
    label[i] = std::tolower(label[i]);

  return label;
}

#endif // defaultModuleLabel_h

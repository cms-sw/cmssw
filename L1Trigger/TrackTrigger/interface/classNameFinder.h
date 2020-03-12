#ifndef classNameFinder_h
#define classNameFinder_h

#include <string>
#include <typeinfo>
#include <cxxabi.h>

template <class T>
std::string classNameFinder(std::string fName) {
  int status2 = 0;
  char *dm2 = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status2);
  std::string s2 = "failed demangle";
  if (status2 == 0) {
    s2 = std::string(dm2);
  }
  return (fName + std::string("< ") + s2 + std::string(" >"));
}

template <class T>
std::string templateNameFinder() {
  int status2 = 0;
  char *dm2 = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status2);
  std::string s2 = "failed demangle";
  if (status2 == 0) {
    s2 = std::string(dm2);
  }
  return s2;
}

#endif

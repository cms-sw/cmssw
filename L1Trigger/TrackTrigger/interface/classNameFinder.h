#ifndef classNameFinder_h
#define classNameFinder_h

#include <string>
#include <typeinfo>
#include <cxxabi.h>

// Prints name of templated class

template <class T>
std::string templateNameFinder() {
  int status = 0;
  char *dm = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  std::string str = "failed demangle";
  if (status == 0 && dm != nullptr) {
    str = std::string(dm);
    std::free(dm);
  }
  return str;
}

template <class T>
std::string classNameFinder(std::string fName) {
  std::string str = templateNameFinder<T>();
  return (fName + std::string("< ") + str + std::string(" >"));
}

#endif

#ifndef binary_ifstream_H
#define binary_ifstream_H

#include <string>
#include <cstdio>
#include "FWCore/Utilities/interface/Visibility.h"

namespace magneticfield::interpolation {
  class binary_ifstream {
  public:
    explicit binary_ifstream(const char* name);
    explicit binary_ifstream(const std::string& name);

    binary_ifstream(const binary_ifstream&) = delete;
    binary_ifstream(binary_ifstream&&);
    binary_ifstream& operator=(const binary_ifstream&) = delete;
    binary_ifstream& operator=(binary_ifstream&&);

    ~binary_ifstream();

    binary_ifstream& operator>>(char& n);
    binary_ifstream& operator>>(unsigned char& n);

    binary_ifstream& operator>>(short& n);
    binary_ifstream& operator>>(unsigned short& n);

    binary_ifstream& operator>>(int& n);
    binary_ifstream& operator>>(unsigned int& n);

    binary_ifstream& operator>>(long& n);
    binary_ifstream& operator>>(unsigned long& n);

    binary_ifstream& operator>>(float& n);
    binary_ifstream& operator>>(double& n);

    binary_ifstream& operator>>(bool& n);
    binary_ifstream& operator>>(std::string& n);

    void close();

    /// stream state checking
    bool good() const;
    bool eof() const;
    bool fail() const;
    bool bad() const;
    bool operator!() const;
    operator bool() const;

    long tellg();
    binary_ifstream& seekg(long);

  private:
    FILE* file_;

    void init(const char* name);
  };
}  // namespace magneticfield::interpolation
#endif

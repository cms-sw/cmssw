#include "MagneticField/Interpolation/interface/binary_ifstream.h"

#include <cstdio>
#include <iostream>

namespace magneticfield::interpolation {
  struct binary_ifstream_error {};

  binary_ifstream::binary_ifstream(const char* name) : file_(nullptr) { init(name); }

  binary_ifstream::binary_ifstream(const std::string& name) : file_(nullptr) { init(name.c_str()); }

  void binary_ifstream::init(const char* name) {
    file_ = fopen(name, "rb");
    if (file_ == nullptr) {
      std::cout << "file " << name << " cannot be opened for reading" << std::endl;
      throw binary_ifstream_error();
    }
  }

  binary_ifstream::binary_ifstream(binary_ifstream&& iOther) : file_(iOther.file_) { iOther.file_ = nullptr; }

  binary_ifstream& binary_ifstream::operator=(binary_ifstream&& iOther) {
    binary_ifstream temp{std::move(iOther)};
    std::swap(file_, temp.file_);
    return *this;
  }

  binary_ifstream::~binary_ifstream() { close(); }
  void binary_ifstream::close() {
    if (file_ != nullptr)
      fclose(file_);
    file_ = nullptr;
  }

  binary_ifstream& binary_ifstream::operator>>(char& n) {
    n = static_cast<char>(fgetc(file_));
    return *this;
  }

  binary_ifstream& binary_ifstream::operator>>(unsigned char& n) {
    n = static_cast<unsigned char>(fgetc(file_));
    return *this;
  }

  binary_ifstream& binary_ifstream::operator>>(short& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }
  binary_ifstream& binary_ifstream::operator>>(unsigned short& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }
  binary_ifstream& binary_ifstream::operator>>(int& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }
  binary_ifstream& binary_ifstream::operator>>(unsigned int& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }

  binary_ifstream& binary_ifstream::operator>>(long& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }
  binary_ifstream& binary_ifstream::operator>>(unsigned long& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }

  binary_ifstream& binary_ifstream::operator>>(float& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }
  binary_ifstream& binary_ifstream::operator>>(double& n) {
    fread(&n, sizeof(n), 1, file_);
    return *this;
  }

  binary_ifstream& binary_ifstream::operator>>(bool& n) {
    n = static_cast<bool>(fgetc(file_));
    return *this;
  }

  binary_ifstream& binary_ifstream::operator>>(std::string& n) {
    unsigned int nchar;
    (*this) >> nchar;
    char* tmp = new char[nchar + 1];
    unsigned int nread = fread(tmp, 1, nchar, file_);
    if (nread != nchar)
      std::cout << "binary_ifstream error: read less then expected " << std::endl;
    n.assign(tmp, nread);
    delete[] tmp;
    return *this;
  }

  bool binary_ifstream::good() const { return !bad() && !eof(); }

  bool binary_ifstream::eof() const { return feof(file_); }

  bool binary_ifstream::fail() const { return file_ == nullptr || ferror(file_) != 0; }

  // don't know the difference between fail() and bad() (yet)
  bool binary_ifstream::bad() const { return fail(); }

  bool binary_ifstream::operator!() const { return fail() || bad() || eof(); }

  //binary_ifstream::operator bool() const {return !fail() && !bad();}

  binary_ifstream::operator bool() const { return good(); }

  long binary_ifstream::tellg() { return ftell(file_); }

  binary_ifstream& binary_ifstream::seekg(long to) {
    //if there is a problem, calling fail() will return true;
    fseek(file_, to, SEEK_SET);
    return *this;
  }

}  // namespace magneticfield::interpolation

#ifndef bstream_iterator_H
#define bstream_iterator_H

#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "binary_ofstream.h"
#include <iterator>
#include "FWCore/Utilities/interface/Visibility.h"

template <typename T>
class bistream_iterator {
public:
  // C++17 compliant iterator definition
  using iterator_category = std::input_iterator_tag;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T*;
  using reference = T&;

  bistream_iterator() : stream_(0) {}

  bistream_iterator(binary_ifstream& s) : stream_(&s) { read(); }

  const T& operator*() const { return value_; }

  const T* operator->() const { return &value_; }

  bistream_iterator& operator++() {
    read();
    return *this;
  }

  bistream_iterator& operator++(int) {
    bistream_iterator tmp;
    read();
    return tmp;
  }

  bool operator==(const bistream_iterator& rhs) { return stream_ == rhs.stream_; }

  bool operator!=(const bistream_iterator& rhs) { return !operator==(rhs); }

private:
  binary_ifstream* stream_;
  T value_;

  void read() {
    if (stream_ != 0) {
      // if (!(*stream_ >> value_)) stream_ = 0;
      if (!(*stream_ >> value_)) {
        stream_ = 0;
        // std::cout << "istream_iterator: stream turned bad, set stream_ to zero" << std::endl;
      }
    }
  }
};

template <typename T>
class dso_internal bostream_iterator : public std::iterator<std::output_iterator_tag, void, void, void, void> {
public:
  bostream_iterator(binary_ofstream& s) : stream_(&s) {}

  bostream_iterator& operator=(const T& t) {
    *stream_ << t;
    return *this;
  }

  bostream_iterator& operator*() { return *this; }
  bostream_iterator& operator++() { return *this; }
  bostream_iterator& operator++(int) { return *this; }

private:
  binary_ofstream* stream_;
};

#endif

#ifndef DataFormats_L1TrackTrigger_TTBV_h
#define DataFormats_L1TrackTrigger_TTBV_h

#include <bitset>
#include <array>
#include <string>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <iostream>

/*! 
 * \class  TTBV
 * \brief  Bit vector used by Track Trigger emulators. Mainly used to convert
 *         integers into arbitrary (within margin) sized two's complement strings.
 * \author Thomas Schuh
 * \date   2020, Jan
 */
class TTBV {
public:
  static constexpr int S_ = 64;  // Frame width of emp infrastructure f/w, max number of bits a TTBV can handle

private:
  bool twos_;           // Two's complement (true) or binary (false)
  int size_;            // number or bits
  std::bitset<S_> bs_;  // underlying storage

public:
  // constructor: default
  TTBV() : twos_(false), size_(0), bs_() {}

  // constructor: double precision (IEEE 754); from most to least significant bit: 1 bit sign + 11 bit binary exponent + 52 bit binary mantisse
  TTBV(const double d) : twos_(false), size_(S_) {
    int index(0);
    const char* c = reinterpret_cast<const char*>(&d);
    for (int iByte = 0; iByte < (int)sizeof(d); iByte++) {
      const std::bitset<std::numeric_limits<unsigned char>::digits> byte(*(c + iByte));
      for (int bit = 0; bit < std::numeric_limits<unsigned char>::digits; bit++)
        bs_[index++] = byte[bit];
    }
  }

  // constructor: unsigned int value
  TTBV(unsigned long long int value, int size) : twos_(false), size_(size), bs_(value) {}

  // constructor: int value
  TTBV(int value, int size, bool twos = false)
      : twos_(twos), size_(size), bs_((!twos || value >= 0) ? value : value + iMax()) {}

  // constructor: double value + precision, biased (floor) representation
  TTBV(double value, double base, int size, bool twos = false) : TTBV((int)std::floor(value / base), size, twos) {}

  // constructor: string
  TTBV(const std::string& str, bool twos = false) : twos_(twos), size_(str.size()), bs_(str) {}

  // constructor: bitset
  TTBV(const std::bitset<S_>& bs, bool twos = false) : twos_(twos), size_(S_), bs_(bs) {}

  // constructor: slice reinterpret sign
  TTBV(const TTBV& ttBV, int begin, int end = 0, bool twos = false) : twos_(twos), size_(begin - end), bs_(ttBV.bs_) {
    bs_ <<= S_ - begin;
    bs_ >>= S_ - begin + end;
  }

  // Two's complement (true) or binary (false)
  bool twos() const { return twos_; }
  // number or bits
  int size() const { return size_; }
  // underlying storage
  const std::bitset<S_>& bs() const { return bs_; }

  // access: single bit
  bool operator[](int pos) const { return bs_[pos]; }
  std::bitset<S_>::reference operator[](int pos) { return bs_[pos]; }

  // access: most significant bit copy
  bool msb() const { return bs_[size_ - 1]; }

  // access: most significant bit reference
  std::bitset<S_>::reference msb() { return bs_[size_ - 1]; }

  // access: members of underlying bitset

  bool all() const { return bs_.all(); }
  bool any() const { return bs_.any(); }
  bool none() const { return bs_.none(); }
  int count() const { return bs_.count(); }

  // operator: comparisons equal
  bool operator==(const TTBV& rhs) const { return bs_ == rhs.bs_; }

  // operator: comparisons not equal
  bool operator!=(const TTBV& rhs) const { return bs_ != rhs.bs_; }

  // operator: boolean and
  TTBV& operator&=(const TTBV& rhs) {
    const int m(std::max(size_, rhs.size()));
    this->resize(m);
    TTBV bv(rhs);
    bv.resize(m);
    bs_ &= bv.bs_;
    return *this;
  }

  // operator: boolean or
  TTBV& operator|=(const TTBV& rhs) {
    const int m(std::max(size_, rhs.size()));
    this->resize(m);
    TTBV bv(rhs);
    bv.resize(m);
    bs_ |= bv.bs_;
    return *this;
  }

  // operator: boolean xor
  TTBV& operator^=(const TTBV& rhs) {
    const int m(std::max(size_, rhs.size()));
    this->resize(m);
    TTBV bv(rhs);
    bv.resize(m);
    bs_ ^= bv.bs_;
    return *this;
  }

  // operator: not
  TTBV operator~() const {
    TTBV bv(*this);
    return bv.flip();
  }

  // reference operator: bit remove right
  TTBV& operator>>=(int pos) {
    bs_ >>= pos;
    size_ -= pos;
    return *this;
  }

  // reference operator: bit remove left
  TTBV& operator<<=(int pos) {
    bs_ <<= S_ - size_ + pos;
    bs_ >>= S_ - size_ + pos;
    size_ -= pos;
    return *this;
  }

  // operator: bit remove left copy
  TTBV operator<<(int pos) const {
    TTBV bv(*this);
    return bv <<= pos;
  }

  // operator: bit remove right copy
  TTBV operator>>(int pos) const {
    TTBV bv(*this);
    return bv >>= pos;
  }

  // reference operator: concatenation
  TTBV& operator+=(const TTBV& rhs) {
    bs_ <<= rhs.size();
    bs_ |= rhs.bs_;
    size_ += rhs.size();
    return *this;
  }

  // operator: concatenation copy
  TTBV operator+(const TTBV& rhs) const {
    TTBV lhs(*this);
    return lhs += rhs;
  }

  // operator: value increment, overflow protected
  TTBV& operator++() {
    bs_ = std::bitset<S_>(bs_.to_ullong() + 1);
    this->resize(size_);
    return *this;
  }

  // manipulation: all bits set to 0
  TTBV& reset() {
    bs_.reset();
    return *this;
  }

  // manipulation: all bits set to 1
  TTBV& set() {
    for (int n = 0; n < size_; n++)
      bs_.set(n);
    return *this;
  }

  // manipulation: all bits flip 1 to 0 and vice versa
  TTBV& flip() {
    for (int n = 0; n < size_; n++)
      bs_.flip(n);
    return *this;
  }

  // manipulation: single bit set to 0
  TTBV& reset(int pos) {
    bs_.reset(pos);
    return *this;
  }

  // manipulation: single bit set to 1
  TTBV& set(int pos) {
    bs_.set(pos);
    return *this;
  }

  // manipulation: multiple bit set to 1
  TTBV& set(std::vector<int> vpos) {
    for (int pos : vpos)
      bs_.set(pos);
    return *this;
  }

  // manipulation: single bit flip 1 to 0 and vice versa
  TTBV& flip(int pos) {
    bs_.flip(pos);
    return *this;
  }

  // manipulation: absolute value of biased twos' complement. Converts twos' complenet into binary.
  TTBV& abs() {
    if (twos_) {
      twos_ = false;
      if (this->msb())
        this->flip();
      size_--;
    }
    return *this;
  }

  // manipulation: resize
  TTBV& resize(int size) {
    bool msb = this->msb();
    if (size > size_) {
      if (twos_)
        for (int n = size_; n < size; n++)
          bs_.set(n, msb);
      size_ = size;
    } else if (size < size_ && size > 0) {
      this->operator<<=(size - size_);
      if (twos_)
        this->msb() = msb;
    }
    return *this;
  }

  // conversion: to string
  std::string str() const { return bs_.to_string().substr(S_ - size_, S_); }

  // conversion: range based to string
  std::string str(int start, int end = 0) const { return this->str().substr(size_ - start, size_ - end); }

  // conversion: to int
  int val() const { return (twos_ && this->msb()) ? (int)bs_.to_ullong() - iMax() : bs_.to_ullong(); }

  // conversion: to int, reinterpret sign
  int val(bool twos) const { return (twos && this->msb()) ? (int)bs_.to_ullong() - iMax() : bs_.to_ullong(); }

  // conversion: range based to int, reinterpret sign
  int val(int start, int end = 0, bool twos = false) const { return TTBV(*this, start, end).val(twos); }

  // conversion: to double for given precision assuming biased (floor) representation
  double val(double base) const { return (this->val() + .5) * base; }

  // conversion: range based to double for given precision assuming biased (floor) representation, reinterpret sign
  double val(double base, int start, int end = 0, bool twos = false) const {
    return (this->val(start, end, twos) + .5) * base;
  }

  // maniplulation and conversion: extracts range based to double reinterpret sign and removes these bits
  double extract(double base, int size, bool twos = false) {
    double val = this->val(base, size, 0, twos);
    this->operator>>=(size);
    return val;
  }

  // maniplulation and conversion: extracts range based to int reinterpret sign and removes these bits
  int extract(int size, bool twos = false) {
    double val = this->val(size, 0, twos);
    this->operator>>=(size);
    return val;
  }

  // manipulation: extracts slice and removes these bits
  TTBV slice(int size, bool twos = false) {
    TTBV ttBV(*this, size, 0, twos);
    this->operator>>=(size);
    return ttBV;
  }

  // range based count of '1's or '0's
  int count(int begin, int end, bool b = true) const {
    int c(0);
    for (int i = begin; i < end; i++)
      if (bs_[i] == b)
        c++;
    return c;
  }

  // position of least significant '1' or '0'
  int plEncode(bool b = true) const {
    for (int e = 0; e < size_; e++)
      if (bs_[e] == b)
        return e;
    return size_;
  }

  // position of most significant '1' or '0'
  int pmEncode(bool b = true) const {
    for (int e = size_ - 1; e > -1; e--)
      if (bs_[e] == b)
        return e;
    return size_;
  }

  // position for n'th '1' or '0' counted from least to most significant bit
  int encode(int n, bool b = true) const {
    int sum(0);
    for (int e = 0; e < size_; e++) {
      if (bs_[e] == b) {
        sum++;
        if (sum == n)
          return e;
      }
    }
    return size_;
  }

  std::vector<int> ids(bool b = true, bool singed = false) const {
    std::vector<int> v;
    v.reserve(bs_.count());
    for (int i = 0; i < size_; i++)
      if (bs_[i] == b)
        v.push_back(singed ? i + size_ / 2 : i);
    return v;
  }

  friend std::ostream& operator<<(std::ostream& os, const TTBV& ttBV) { return os << ttBV.str(); }

private:
  // look up table initializer for powers of 2
  constexpr std::array<unsigned long long int, S_> powersOfTwo() const {
    std::array<unsigned long long int, S_> lut = {};
    for (int i = 0; i < S_; i++)
      lut[i] = std::pow(2, i);
    return lut;
  }

  // returns 2 ** size_
  unsigned long long int iMax() const {
    static const std::array<unsigned long long int, S_> lut = powersOfTwo();
    return lut[size_];
  }
};

#endif

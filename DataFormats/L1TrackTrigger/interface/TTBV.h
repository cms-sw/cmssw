#ifndef __L1TrackTrigger_TTBV__
#define __L1TrackTrigger_TTBV__

#include <bitset>
#include <string>
#include <algorithm>

/*! 
 * \class  TTBV
 * \brief  Bit vector used by Track Trigger emulators. Mainly used to convert
 *         integers into arbitrary (within margin) sized two's complement string.
 * \author Thomas Schuh
 * \date   2020, Jan
 */
class TTBV {
public:
  static constexpr int S = 64;  // Frame width of emp infrastructure f/w, max number of bits a TTBV can handle

private:
  bool signed_;        // Two's complement (true) or binary (false)
  int size_;           // number or bits
  std::bitset<S> bs_;  // underlying storage

public:
  // constructor: default
  TTBV() : signed_(false), size_(S), bs_(std::bitset<S>(0)) {}

  // constructor: double precision (IEEE 754); from most to least significant bit: 1 bit sign + 11 bit binary exponent + 52 bit binary mantisse
  TTBV(const double& d) : signed_(false), size_(S) {
    std::string str;
    for (int i = (int)sizeof(d) - 1; i >= 0; i--) {
      const char* c = reinterpret_cast<const char*>(&d) + i;
      str += std::bitset<8>(*c).to_string();
    }
    bs_ = std::bitset<S>(str);
  }

  // constructor: unsigned int value
  TTBV(const unsigned long long int& value, const int& size)
      : signed_(false), size_(size), bs_(std::bitset<S>(value)) {}

  // constructor: int value
  TTBV(const int& value, const int& size, const bool& Signed = false)
      : signed_(Signed),
        size_(size),
        bs_(std::bitset<S>((!Signed || value >= 0) ? value : value + std::pow(2, size_))) {}

  // constructor: double value + precision, biased (floor) representation
  TTBV(const double& value, const double& base, const int& size, const bool& Signed = false)
      : TTBV((int)std::floor(value / base), size, Signed) {}

  // constructor: string
  TTBV(const std::string& str, const bool& Signed = false)
      : signed_(Signed), size_(str.size()), bs_(std::bitset<S>(str)) {}

  // constructor: bitset
  TTBV(const std::bitset<S>& bs, const bool& Signed = false) : signed_(Signed), size_(S), bs_(std::bitset<S>(bs)) {}

  // access: data members
  bool Signed() const { return signed_; }
  int size() const { return size_; }
  std::bitset<S> bs() const { return bs_; }

  // access: single bit
  bool operator[](const int& pos) const { return bs_[pos]; }
  std::bitset<S>::reference operator[](const int& pos) { return bs_[pos]; }

  // access: most significant bit
  bool msb() const { return bs_[size_ - 1]; }
  std::bitset<S>::reference msb() { return bs_[size_ - 1]; }

  // access: members of underlying bitset
  bool all() const { return bs_.all(); }
  bool any() const { return bs_.any(); }
  bool none() const { return bs_.none(); }
  int count() const { return bs_.count(); }

  // operator: comparisons
  bool operator==(const TTBV& rhs) const { return bs_ == rhs.bs_; }
  bool operator!=(const TTBV& rhs) const { return bs_ != rhs.bs_; }

  // operator: boolean and, or, xor, not
  TTBV& operator&=(const TTBV& rhs) {
    const int m(std::max(size_, rhs.size()));
    this->resize(m);
    TTBV bv(rhs);
    bv.resize(m);
    bs_ &= bv.bs_;
    return *this;
  }
  TTBV& operator|=(const TTBV& rhs) {
    const int m(std::max(size_, rhs.size()));
    this->resize(m);
    TTBV bv(rhs);
    bv.resize(m);
    bs_ |= bv.bs_;
    return *this;
  }
  TTBV& operator^=(const TTBV& rhs) {
    const int m(std::max(size_, rhs.size()));
    this->resize(m);
    TTBV bv(rhs);
    bv.resize(m);
    bs_ ^= bv.bs_;
    return *this;
  }
  TTBV operator~() const {
    TTBV bv(*this);
    return bv.flip();
  }

  // operator: bit shifts
  TTBV& operator>>=(const int& pos) {
    bs_ >>= pos;
    size_ -= pos;
    return *this;
  }
  TTBV& operator<<=(const int& pos) {
    bs_ <<= S - size_ + pos;
    bs_ >>= S - size_ + pos;
    size_ -= pos;
    return *this;
  }
  TTBV operator<<(const int& pos) const {
    TTBV bv(*this);
    return bv >>= pos;
  }
  TTBV operator>>(const int& pos) const {
    TTBV bv(*this);
    return bv <<= pos;
  }

  // operator: concatenation
  TTBV& operator+=(const TTBV& rhs) {
    bs_ <<= rhs.size();
    bs_ |= rhs.bs_;
    size_ += rhs.size();
    return *this;
  }
  TTBV operator+(const TTBV& rhs) const {
    TTBV lhs(*this);
    return lhs += rhs;
  }

  // operator: value increment, overflow protected
  TTBV& operator++() {
    bs_ = std::bitset<S>(bs_.to_ullong() + 1);
    this->resize(size_);
    return *this;
  }

  // manipulation: all bits
  TTBV& reset() {
    bs_.reset();
    return *this;
  }
  TTBV& set() {
    for (int n = 0; n < size_; n++)
      bs_.set(n);
    return *this;
  }
  TTBV& flip() {
    for (int n = 0; n < size_; n++)
      bs_.flip(n);
    return *this;
  }

  // manipulation: single bit
  TTBV& reset(const int& pos) {
    bs_.reset(pos);
    return *this;
  }
  TTBV& set(const int& pos) {
    bs_.set(pos);
    return *this;
  }
  TTBV& flip(const int& pos) {
    bs_.flip(pos);
    return *this;
  }

  // manipulation: biased absolute value
  TTBV& abs() {
    if (signed_) {
      signed_ = false;
      if (this->msb()) {
        this->flip();
        this->operator++();
      }
      size_--;
    }
    return *this;
  }

  // manipulation: resize
  TTBV& resize(const int& size) {
    bool msb = this->msb();
    if (size > size_) {
      if (signed_)
        for (int n = size_; n < size; n++)
          bs_.set(n, msb);
      size_ = size;
    } else if (size < size_ && size > 0) {
      this->operator<<=(size - size_);
      if (signed_)
        this->msb() = msb;
    }
    return *this;
  }

  // conversion: to string
  std::string str() const { return bs_.to_string().substr(S - size_, S); }

  // conversion: range based to string
  std::string str(const int& start, const int& end = 0) const { return this->str().substr(size_ - start, size_ - end); }

  // conversion: to int
  int val() const { return (signed_ && this->msb()) ? (int)bs_.to_ullong() - std::pow(2, size_) : bs_.to_ullong(); }

  // conversion: to int, reinterpret sign
  int val(const bool Signed) const {
    return (Signed && this->msb()) ? (int)bs_.to_ullong() - std::pow(2, size_) : bs_.to_ullong();
  }

  // conversion: range based to int, reinterpret sign
  int val(const int& start, const int& end = 0, const bool& Signed = false) const {
    return TTBV(this->str(start, end), Signed).val();
  }

  // conversion: to double for given precision assuming biased (floor) representation
  double val(const double& base) const { return (this->val() + .5) * base; }

  // range based count of '1's or '0's
  int count(const int& begin, const int& end, const bool& b = true) const {
    int c(0);
    for (int i = begin; i < end; i++)
      if (bs_[i] == b)
        c++;
    return c;
  }

  // position of least significant '1' or '0'
  int plEncode(const bool& b = true) const {
    for (int e = 0; e < size_; e++)
      if (bs_[e] == b)
        return e;
    return size_;
  }

  // position of most significant '1' or '0'
  int pmEncode(const bool& b = true) const {
    for (int e = size_ - 1; e > -1; e--)
      if (bs_[e] == b)
        return e;
    return size_;
  }
};

#endif
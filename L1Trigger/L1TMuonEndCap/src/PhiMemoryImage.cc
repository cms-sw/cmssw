#include "L1Trigger/L1TMuonEndCap/interface/PhiMemoryImage.h"

#include <stdexcept>
#include <iostream>
#include <bitset>

#define UINT64_BITS 64


PhiMemoryImage::PhiMemoryImage() {
  reset();
}

PhiMemoryImage::~PhiMemoryImage() {

}

PhiMemoryImage::PhiMemoryImage(const PhiMemoryImage& other) {
  std::copy(&(other._buffer[0][0]), &(other._buffer[0][0]) + (_layers*_units), &(_buffer[0][0]));

  _straightness = other._straightness;
}

PhiMemoryImage::PhiMemoryImage(PhiMemoryImage&& other) noexcept : PhiMemoryImage() {
  swap(other);
}

// Copy-and-swap idiom
PhiMemoryImage& PhiMemoryImage::operator=(PhiMemoryImage other) {
  swap(other);
  return *this;
}

void PhiMemoryImage::swap(PhiMemoryImage& other) {
  std::swap_ranges(&(other._buffer[0][0]), &(other._buffer[0][0]) + (_layers*_units), &(_buffer[0][0]));

  std::swap(other._straightness, _straightness);
}

void PhiMemoryImage::reset() {
  std::fill(&(_buffer[0][0]), &(_buffer[0][0]) + (_layers*_units), 0);

  _straightness = 0;
}

void PhiMemoryImage::set_bit(unsigned int layer, unsigned int bit) {
  check_input(layer, bit);
  value_type unit = bit / UINT64_BITS;
  value_type mask = (1ul << (bit % UINT64_BITS));
  _buffer[layer][unit] |= mask;
}

void PhiMemoryImage::clear_bit(unsigned int layer, unsigned int bit) {
  check_input(layer, bit);
  value_type unit = bit / UINT64_BITS;
  value_type mask = (1ul << (bit % UINT64_BITS));
  _buffer[layer][unit] &= ~mask;
}

bool PhiMemoryImage::test_bit(unsigned int layer, unsigned int bit) const {
  check_input(layer, bit);
  value_type unit = bit / UINT64_BITS;
  value_type mask = (1ul << (bit % UINT64_BITS));
  return _buffer[layer][unit] & mask;
}

void PhiMemoryImage::set_word(unsigned int layer, unsigned int unit, value_type value) {
  check_input(layer, unit*UINT64_BITS);
  _buffer[layer][unit] = value;
}

PhiMemoryImage::value_type PhiMemoryImage::get_word(unsigned int layer, unsigned int unit) const {
  check_input(layer, unit*UINT64_BITS);
  return _buffer[layer][unit];
}

void PhiMemoryImage::check_input(unsigned int layer, unsigned int bit) const {
  if (layer >= _layers) {
    char what[128];
    snprintf(what, sizeof(what), "layer (which is %u) >= _layers (which is %u)", layer, _layers);
    throw std::out_of_range(what);
  }

  unsigned int unit = bit / UINT64_BITS;
  if (unit >= _units) {
    char what[128];
    snprintf(what, sizeof(what), "unit (which is %u) >= _units (which is %u)", unit, _units);
    throw std::out_of_range(what);
  }
}

// See https://en.wikipedia.org/wiki/Circular_shift#Implementing_circular_shifts
// return (val << len) | ((unsigned) val >> (-len & (sizeof(INT) * CHAR_BIT - 1)));
void PhiMemoryImage::rotl(unsigned int n) {
  if (n >= _units*UINT64_BITS)
    return;

  value_type tmp[_layers][_units];
  std::copy(&(_buffer[0][0]), &(_buffer[0][0]) + (_layers*_units), &(tmp[0][0]));

  const unsigned int mask = UINT64_BITS - 1;
  const unsigned int n1 = n % UINT64_BITS;
  const unsigned int n2 = _units - (n / UINT64_BITS);
  const unsigned int n3 = (n1 == 0) ? n2+1 : n2;

  unsigned int i = 0, j = 0, j_curr = 0, j_next = 0;
  for (i = 0; i < _layers; ++i) {
    for (j = 0; j < _units; ++j) {
      // if n2 == 0:
      //   j_curr = 0, 1, 2
      //   j_next = 2, 0, 1
      // if n2 == 1:
      //   j_curr = 2, 0, 1
      //   j_next = 1, 2, 0
      j_curr = (n2+j) % _units;
      j_next = (n3+j+_units-1) % _units;
      _buffer[i][j] = (tmp[i][j_curr] << n1) | (tmp[i][j_next] >> (-n1 & mask));
    }
  }
}

void PhiMemoryImage::rotr(unsigned int n) {
  if (n >= _units*UINT64_BITS)
    return;

  value_type tmp[_layers][_units];
  std::copy(&(_buffer[0][0]), &(_buffer[0][0]) + (_layers*_units), &(tmp[0][0]));

  const unsigned int mask = UINT64_BITS - 1;
  const unsigned int n1 = n % UINT64_BITS;
  const unsigned int n2 = n / UINT64_BITS;
  const unsigned int n3 = (n1 == 0) ? n2+_units-1 : n2;

  unsigned int i = 0, j = 0, j_curr = 0, j_next = 0;
  for (i = 0; i < _layers; ++i) {
    for (j = 0; j < _units; ++j) {
      // if n2 == 0:
      //   j_curr = 0, 1, 2
      //   j_next = 1, 2, 0
      // if n2 == 1:
      //   j_curr = 2, 0, 1
      //   j_next = 0, 1, 2
      j_curr = (n2+j)% _units;
      j_next = (n3+j+1) % _units;
      _buffer[i][j] = (tmp[i][j_curr] >> n1) | (tmp[i][j_next] << (-n1 & mask));
    }
  }
}

unsigned int PhiMemoryImage::op_and(const PhiMemoryImage& other) const {
  static_assert((_layers == 4 && _units == 3), "This function assumes (_layers == 4 && _units == 3)");

  // Unroll
  bool b_st1 = (_buffer[0][0] & other._buffer[0][0]) ||
               (_buffer[0][1] & other._buffer[0][1]) ||
               (_buffer[0][2] & other._buffer[0][2]);
  bool b_st2 = (_buffer[1][0] & other._buffer[1][0]) ||
               (_buffer[1][1] & other._buffer[1][1]) ||
               (_buffer[1][2] & other._buffer[1][2]);
  bool b_st3 = (_buffer[2][0] & other._buffer[2][0]) ||
               (_buffer[2][1] & other._buffer[2][1]) ||
               (_buffer[2][2] & other._buffer[2][2]);
  bool b_st4 = (_buffer[3][0] & other._buffer[3][0]) ||
               (_buffer[3][1] & other._buffer[3][1]) ||
               (_buffer[3][2] & other._buffer[3][2]);

  //   bit 0: st3 or st4 hit
  //   bit 1: st2 hit
  //   bit 2: st1 hit
  unsigned int ly = (b_st1 << 2) | (b_st2 << 1) | (b_st3 << 0) | (b_st4 << 0);
  return ly;
}

void PhiMemoryImage::print(std::ostream& out) const {
  constexpr int N = 160;
  out << std::bitset<N-128>(_buffer[3][2]) << std::bitset<128-64>(_buffer[3][1]) << std::bitset<64>(_buffer[3][0]) << std::endl;
  out << std::bitset<N-128>(_buffer[2][2]) << std::bitset<128-64>(_buffer[2][1]) << std::bitset<64>(_buffer[2][0]) << std::endl;
  out << std::bitset<N-128>(_buffer[1][2]) << std::bitset<128-64>(_buffer[1][1]) << std::bitset<64>(_buffer[1][0]) << std::endl;
  out << std::bitset<N-128>(_buffer[0][2]) << std::bitset<128-64>(_buffer[0][1]) << std::bitset<64>(_buffer[0][0]);
}

// _____________________________________________________________________________
// Output streams
std::ostream& operator<<(std::ostream& o, const PhiMemoryImage& patt) {
  patt.print(o);
  return o;
}

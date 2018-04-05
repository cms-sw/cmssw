#ifndef L1TMuonEndCap_PhiMemoryImage_h
#define L1TMuonEndCap_PhiMemoryImage_h

#include <cstdint>
#include <iosfwd>

// Originally written by Ivan Furic and Matt Carver (Univ of Florida)


class PhiMemoryImage {
public:
  typedef uint64_t value_type;

  PhiMemoryImage();
  ~PhiMemoryImage();

  // Copy constructor, move constructor and copy assignment
  PhiMemoryImage(const PhiMemoryImage& other);
  PhiMemoryImage(PhiMemoryImage&& other) noexcept;
  PhiMemoryImage& operator=(PhiMemoryImage other);

  void swap(PhiMemoryImage& other);

  void reset();

  void set_bit(unsigned int layer, unsigned int bit);

  void clear_bit(unsigned int layer, unsigned int bit);

  bool test_bit(unsigned int layer, unsigned int bit) const;

  void set_word(unsigned int layer, unsigned int unit, value_type value);

  value_type get_word(unsigned int layer, unsigned int unit) const;

  void set_straightness(int s) { _straightness = s; }

  int get_straightness() const { return _straightness; }

  // Left rotation by n bits
  void rotl(unsigned int n);

  // Right rotation by n bits
  void rotr(unsigned int n);

  // Kind of like AND operator
  // It returns a layer code which encodes
  //   bit 0: st3 or st4 hit
  //   bit 1: st2 hit
  //   bit 2: st1 hit
  unsigned int op_and(const PhiMemoryImage& other) const;

  void print(std::ostream& out) const;

private:
  void check_input(unsigned int layer, unsigned int bit) const;

  // Num of layers
  //   [0,1,2,3] --> [st1,st2,st3,st4]
  static const unsigned int _layers = 4;

  // Num of value_type allocated per layer
  //   3 * 64 bits = 192 bits
  static const unsigned int _units = 3;

  // Hits in non-key stations
  value_type _buffer[_layers][_units];

  int _straightness;
};

// _____________________________________________________________________________
// Output streams
std::ostream& operator<<(std::ostream& o, const PhiMemoryImage& p);

#endif

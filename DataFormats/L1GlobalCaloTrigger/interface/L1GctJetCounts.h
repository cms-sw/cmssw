#ifndef L1GCTJETCOUNTS_H
#define L1GCTJETCOUNTS_H

#include <vector>
#include <ostream>
#include <stdint.h>

///
/// \class L1GctJetCounts
/// 
/// \author: Jim Brooke
///
/// Class to store the GCT jet count output
/// 


class L1GctJetCounts {

 public:

  /// static maximum number of jet counts
  /// This can be up to 12 but we use some of the
  /// available bandwidth for other information.
  static const unsigned MAX_TOTAL_COUNTS;
  static const unsigned MAX_TRUE_COUNTS;

  /// default constructor
  L1GctJetCounts();

  /// Constructor for unpacking.
  /*! Expects three 5-bit jet counts in bits 14:0, and then
   *  three more 5-bit jet counts in bits 30:16 for both of
   *  the arguments; this is because in the raw format bit
   *  31 is a BC0 flag, and bit 15 is always 1. Thus, jet
   *  count 0 should be in bits 4:0 of the data0 argument. */
  L1GctJetCounts(uint32_t data0, uint32_t data1);

  L1GctJetCounts(uint32_t data0, uint32_t data1, int16_t bx);

  /// constructor for emulator
  L1GctJetCounts(const std::vector<unsigned>& counts);

  L1GctJetCounts(const std::vector<unsigned>& counts, int16_t bx);

  /// destructor
  virtual ~L1GctJetCounts();

  /// name method
  std::string name() const { return "JetCounts"; }

  /// empty method
  bool empty() const { return false; }

  /// get raw word 0
  uint32_t raw0() const { return m_data0; }

  /// get raw word 1
  uint32_t raw1() const { return m_data1; }

  /// get count by index
  unsigned count(unsigned i) const;

  /// get individual counts (for use with FWLite)
  unsigned count00() const { return (MAX_TRUE_COUNTS<1 ?  0 : count(0)); }
  unsigned count01() const { return (MAX_TRUE_COUNTS<2 ?  0 : count(1)); }
  unsigned count02() const { return (MAX_TRUE_COUNTS<3 ?  0 : count(2)); }
  unsigned count03() const { return (MAX_TRUE_COUNTS<4 ?  0 : count(3)); }
  unsigned count04() const { return (MAX_TRUE_COUNTS<5 ?  0 : count(4)); }
  unsigned count05() const { return (MAX_TRUE_COUNTS<6 ?  0 : count(5)); }
  unsigned count06() const { return (MAX_TRUE_COUNTS<7 ?  0 : count(6)); }
  unsigned count07() const { return (MAX_TRUE_COUNTS<8 ?  0 : count(7)); }
  unsigned count08() const { return (MAX_TRUE_COUNTS<9 ?  0 : count(8)); }
  unsigned count09() const { return (MAX_TRUE_COUNTS<10 ? 0 : count(9)); }
  unsigned count10() const { return (MAX_TRUE_COUNTS<11 ? 0 : count(10)); }
  unsigned count11() const { return (MAX_TRUE_COUNTS<12 ? 0 : count(11)); }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator
  int operator==(const L1GctJetCounts& c) const { return (m_data0==c.raw0() && m_data1==c.raw1()); }

  /// inequality operator
  int operator!=(const L1GctJetCounts& c) const { return !(*this == c); }

 private:

  uint32_t m_data0;
  uint32_t m_data1;
  int16_t m_bx;

};

std::ostream& operator<<(std::ostream& s, const L1GctJetCounts& c);

#endif

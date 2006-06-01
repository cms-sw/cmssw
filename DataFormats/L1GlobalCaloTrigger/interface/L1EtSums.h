#ifndef L1GCTETSUMS_H
#define L1GCTETSUMS_H

#include <boost/cstdint.hpp>
#include <ostream>

/*! \file L1GctEtSums.h
 * \Header file for the GCT energy sum output
 * 
 * \author: Jim Brooke
 *
 */


class L1EtTotal {
 public:
  L1EtTotal();
  L1EtTotal(uint16_t data);
  L1EtTotal(unsigned et, bool oflow);
  ~L1EtTotal();

  ///
  /// get the data
  uint16_t raw() const { return m_data; }
  ///
  /// get the Et
  unsigned et() const { return m_data & 0xfff; }
  ///
  /// get the overflow
  bool overflow() const { return (m_data & 0x1000)!=0; }

 Private:

  uint16_t m_data;

};

class L1EtHad {
 public:
  L1EtHad();
  L1EtHad(uint16_t data);
  L1EtHad(unsigned et, bool oflow);
  ~L1EtHad();

  ///
  /// get the data
  uint16_t raw() const { return m_data; }
  ///
  /// get the Et
  unsigned et() const { return m_data & 0xfff; }
  ///
  /// get the overflow
  bool overFlow() const { return (m_data & 0x1000)!=0; }

 private:

  uint16_t m_data;

};

class L1EtMiss {
 public:
  L1EtMiss();
  L1EtMiss(uint32_t data);
  L1EtMiss(unsigned et, unsigned phi, bool oflow);
  ~L1EtMiss();

  ///
  /// get the data
  uint32_t raw() const { return m_data; }
  ///
  /// get the magnitude
  unsigned et() const { return m_data & 0xfff; }
  ///
  /// get the overflow
  bool overFlow() const { return (m_data & 0x1000)!=0; }
  ///
  /// get the Et
  unsigned phi() const { return (m_data>>13) & 0x7f; }

 private:

  uint32_t m_data;

};

#endif

#ifndef L1GCTETSUMS_H
#define L1GCTETSUMS_H

#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1GlobalTrigger/interface/L1TriggerObject.h"

/*! \file L1GctEtSums.h
 * \Header file for the GCT energy sum output
 * 
 * \author: Jim Brooke
 *
 */


class L1GctEtTotal {
 public:
  L1GctEtTotal();
  L1GctEtTotal(uint16_t data);
  L1GctEtTotal(unsigned et, bool oflow);
  virtual ~L1GctEtTotal();

  /// name method
  std::string name() const { return "EtTotal"; }

  /// empty method (= false; total Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  /// get the Et
  unsigned et() const { return m_data & 0xfff; }

  /// get the overflow
  bool overflow() const { return (m_data & 0x1000)!=0; }

 private:

  uint16_t m_data;

};

class L1GctEtHad {
 public:
  L1GctEtHad();
  L1GctEtHad(uint16_t data);
  L1GctEtHad(unsigned et, bool oflow);
  virtual ~L1GctEtHad();

  /// name method
  std::string name() const { return "EtHad"; }

  /// empty method (= false; hadronic Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint16_t raw() const { return m_data; }

  /// get the Et
  unsigned et() const { return m_data & 0xfff; }

  /// get the overflow
  bool overFlow() const { return (m_data & 0x1000)!=0; }

 private:

  uint16_t m_data;

};

class L1GctEtMiss {
 public:
  L1GctEtMiss();
  L1GctEtMiss(uint32_t data);
  L1GctEtMiss(unsigned et, unsigned phi, bool oflow);
  virtual ~L1GctEtMiss();

  /// name method
  std::string name() const { return "EtMiss"; }

  /// empty method (= false; missing Et is always calculated)
  bool empty() const { return false; }

  /// get the data
  uint32_t raw() const { return m_data; }

  /// get the magnitude
  unsigned et() const { return m_data & 0xfff; }

  /// get the overflow
  bool overFlow() const { return (m_data & 0x1000)!=0; }

  /// get the Et
  unsigned phi() const { return (m_data>>13) & 0x7f; }

 private:

  uint32_t m_data;

};

#endif

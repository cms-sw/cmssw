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


class L1GctEtTotal {
 public:
  L1GctEtTotal();
  L1GctEtTotal(uint16_t data);
  L1GctEtTotal(int et, bool oflow);
  ~L1GctEtTotal();

  ///
  /// get the data
  uint16_t raw() const { return theEtTotal; }
  ///
  /// get the Et
  int et() const { return theEtTotal & 0xfff; }
  ///
  /// get the overflow
  bool overflow() const { return (theEtTotal & 0x1000)!=0; }

 private:

  uint16_t theEtTotal;

};

class L1GctEtHad {
 public:
  L1GctEtHad();
  L1GctEtHad(uint16_t data);
  L1GctEtHad(int et, bool oflow);
  ~L1GctEtHad();

  ///
  /// get the data
  uint16_t raw() const { return theEtHad; }
  ///
  /// get the Et
  int et() const { return theEtHad & 0xfff; }
  ///
  /// get the overflow
  bool overflow() const { return (theEtHad & 0x1000)!=0; }

 private:

  uint16_t theEtHad;

};

class L1GctEtMiss {
 public:
  L1GctEtMiss();
  L1GctEtMiss(uint32_t data);
  L1GctEtMiss(int et, int phi, bool oflow);
  ~L1GctEtMiss();

  ///
  /// get the data
  uint32_t raw() const { return theEtMiss; }
  ///
  /// get the magnitude
  int et() const { return theEtMiss & 0xfff; }
  ///
  /// get the overflow
  bool overflow() const { return (theEtMiss & 0x1000)!=0; }
  ///
  /// get the Et
  int phi() const { return (theEtMiss>>13) & 0x7f; }

 private:

  uint16_t theEtMiss;

};

#endif

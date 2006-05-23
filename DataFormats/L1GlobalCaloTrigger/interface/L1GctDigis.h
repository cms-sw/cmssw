#ifndef L1GCTDIGIS_H
#define L1GCTDIGIS_H

#include <boost/cstdint.hpp>
#include <ostream>

/*! \file L1GctDigis.h
 * \Header file for the Gct
 *  electron and jet candidates
 * 
 * \author: Jim Brooke
 *
 */

class L1GctCand
{
public:
  L1GctCand();
  L1GctCand(uint16_t data);
  L1GctCand(int rank, int phi, int eta);
  ~L1GctCand();

  ///
  /// get the raw data
  uint16_t raw() const { return m_data; }
  ///
  /// get rank bits
  int rank() const { return m_data & 0x3f; }
  ///
  /// get phi bits
  int phi() const { return (m_data>>6) & 0x1f; }
  ///
  /// get eta bits
  int eta() const { return (m_data>>11) & 0xf; }

private:

  uint16_t m_data;

};

class L1GctIsoEm : public L1GctCand {
public:
  L1GctIsoEm();
  L1GctIsoEm(uint16_t data);
  L1GctIsoEm(int rank, int phi, int eta);
  ~L1GctIsoEm();
 };

class L1GctNonIsoEm : public L1GctCand {
public:
  L1GctNonIsoEm();
  L1GctNonIsoEm(uint16_t data);
  L1GctNonIsoEm(int rank, int phi, int eta);
  ~L1GctNonIsoEm();
 };

class L1GctCenJet : public L1GctCand {
public:
  L1GctCenJet();
  L1GctCenJet(uint16_t data);
  L1GctCenJet(int rank, int phi, int eta);
  ~L1GctCenJet();
 };

class L1GctForJet : public L1GctCand {
public:
  L1GctForJet();
  L1GctForJet(uint16_t data);
  L1GctForJet(int rank, int phi, int eta);
  ~L1GctForJet();
 };

class L1GctTauJet : public L1GctCand {
public:
  L1GctTauJet();
  L1GctTauJet(uint16_t data);
  L1GctTauJet(int rank, int phi, int eta);
  ~L1GctTauJet();
 };


std::ostream& operator<<(std::ostream& s, const L1GctCand& cand);

std::ostream& operator<<(std::ostream& s, const L1GctIsoEm& cand);

std::ostream& operator<<(std::ostream& s, const L1GctNonIsoEm& cand);

std::ostream& operator<<(std::ostream& s, const L1GctCenJet& cand);

std::ostream& operator<<(std::ostream& s, const L1GctForJet& cand);

std::ostream& operator<<(std::ostream& s, const L1GctTauJet& cand);



#endif 

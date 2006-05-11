#ifndef L1GCTCANDIDATES_H
#define L1GCTCANDIDATES_H

#include <boost/cstdint.hpp>
#include <ostream>

/*! \file L1GctCandidates.h
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
  uint16_t raw() const { return theCand; }
  ///
  /// get rank bits
  int rank() const { return theCand & 0x3f; }
  ///
  /// get phi bits
  int phi() const { return (theCand>>6) & 0x1f; }
  ///
  /// get eta bits
  int eta() const { return (theCand>>11) & 0xf; }

private:

  uint16_t theCand;

};

std::ostream& operator<<(std::ostream& s, const L1GctCand& cand);

class L1GctIsoEmCand : public L1GctCand {
public:
  L1GctIsoEmCand();
  L1GctIsoEmCand(uint16_t data);
  L1GctIsoEmCand(int rank, int phi, int eta);
  ~L1GctIsoEmCand();
 };

class L1GctNonIsoEmCand : public L1GctCand {
public:
  L1GctNonIsoEmCand();
  L1GctNonIsoEmCand(uint16_t data);
  L1GctNonIsoEmCand(int rank, int phi, int eta);
  ~L1GctNonIsoEmCand();
 };

class L1GctCenJetCand : public L1GctCand {
public:
  L1GctCenJetCand();
  L1GctCenJetCand(uint16_t data);
  L1GctCenJetCand(int rank, int phi, int eta);
  ~L1GctCenJetCand();
 };

class L1GctForJetCand : public L1GctCand {
public:
  L1GctForJetCand();
  L1GctForJetCand(uint16_t data);
  L1GctForJetCand(int rank, int phi, int eta);
  ~L1GctForJetCand();
 };

class L1GctTauJetCand : public L1GctCand {
public:
  L1GctTauJetCand();
  L1GctTauJetCand(uint16_t data);
  L1GctTauJetCand(int rank, int phi, int eta);
  ~L1GctTauJetCand();
 };



#endif 

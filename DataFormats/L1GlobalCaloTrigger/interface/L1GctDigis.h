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

/*! \class L1GctCand
 * \brief Base class for EM and jet candidates, for convenience (code re-use!)
 *
 */

class L1GctCand
{
public:
  /// default constructor (required for vector initialisation etc.)
  L1GctCand();

  /// construct from raw data
  L1GctCand(uint16_t data);

  /// construct from separate rank, eta, phi
  L1GctCand(int rank, int phi, int eta);

  /// destruct
  ~L1GctCand();
  
  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  int rank() const { return m_data & 0x3f; }
  
  /// get phi bits
  int phi() const { return (m_data>>6) & 0x1f; }
  
  /// get eta bits
  int eta() const { return (m_data>>11) & 0xf; }


private:

  uint16_t m_data;

};

/*! \class L1GctEmCand
 * \brief Level-1 Trigger EM candidate
 *
 */

class L1GctEmCand : public L1GctCand {
public:
  /// default constructor (for vector initialisation etc.)
  L1GctEmCand();
  /// construct from raw data
  L1GctEmCand(uint16_t data);
  /// construct from rank, eta, phi, isolation and RCT crate #
  L1GctEmCand(int rank, int phi, int eta, bool iso, unsigned rctCrate);
   /// destructor
 ~L1GctEmCand();
 
  /// which stream did this come from
  bool isolated() const { return m_iso; }

  /// which RCT crate did this came from
  int rctCrate() const { return m_rctCrate; }

 private:

  bool m_iso;
  unsigned m_rctCrate;

 };


/*! \class L1GctJetCand
 * \brief Level-1 Trigger jet candidate
 *
 */

class L1GctJetCand : public L1GctCand {
public:
  /// default constructor (for vector initialisation etc.)
  L1GctJetCand();
  /// construct from raw data
  L1GctJetCand(uint16_t data);
  /// construct from rank, eta, phi
  L1GctJetCand(int rank, int phi, int eta, bool isTau, bool isFor);
  /// destructor
  ~L1GctJetCand();

  /// check if this is a tau
  bool isTau() const { return m_isTau; }

  /// check if this is a forward jet
  bool isFor() const { return m_isFor; }

 private:

  bool m_isTau;
  bool m_isFor;

 };


std::ostream& operator<<(std::ostream& s, const L1GctCand& cand);

std::ostream& operator<<(std::ostream& s, const L1GctEmCand& cand);

std::ostream& operator<<(std::ostream& s, const L1GctJetCand& cand);


#endif 

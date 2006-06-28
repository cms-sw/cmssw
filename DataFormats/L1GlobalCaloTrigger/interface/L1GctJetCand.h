#ifndef L1GCTJETCAND_H
#define L1GCTJETCAND_H

#include <boost/cstdint.hpp>
#include <ostream>

/*! \class L1GctJetCand
 * \brief Level-1 Trigger jet candidate
 *
 */

/*! \author Jim Brooke, Sridhara Dasu
 *  \date June 2006
 */


class L1GctJetCand {
public:
  /// default constructor (for vector initialisation etc.)
  L1GctJetCand();

  /// construct from raw data
  L1GctJetCand(uint16_t data, bool isTau, bool isFor);

  /// construct from rank, eta, phi
  L1GctJetCand(unsigned rank, int phi, int eta, bool isTau, bool isFor);

  /// destructor
  ~L1GctJetCand();

  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  unsigned rank() const { return m_data & 0x3f; }
  
  /// which half of the detector?
  int zside() const { return (m_data&0x8)?(1):(-1) ; }

  /// get magnitude of eta
  int ietaAbs() const { return (m_data>>6) & 0x7; }

  /// get eta index
  int ieta() const { return zside()*ietaAbs(); }

  /// get phi index
  int iphi() const { return (m_data>>10) & 0x1f; }

  /// get eta bits
  int level1EtaIndex() const { return ieta(); }

  /// get phi bits
  int level1PhiIndex() const { return iphi(); }

  /// check if this is a central jet
  bool isCentral() const { return (!m_isTau) && (!m_isFor); }

  /// check if this is a tau
  bool isTau() const { return m_isTau; }

  /// check if this is a forward jet
  bool isForward() const { return m_isFor; }

 private:

  uint16_t m_data;
  bool m_isTau;
  bool m_isFor;

 };

std::ostream& operator<<(std::ostream& s, const L1GctJetCand& cand);

#endif

#ifndef L1GCTEMCAND_H
#define L1GCTEMCAND_H

#include <boost/cstdint.hpp>
#include <ostream>

/*! \class L1GctEmCand
 * \brief Level-1 Trigger EM candidate at output of GCT
 *
 */

/*! \author Jim Brooke
 *  \date June 2006
 */


class L1GctEmCand {
public:

  /// default constructor (for vector initialisation etc.)
  L1GctEmCand();

  /// construct from raw data
  L1GctEmCand(uint16_t data, bool iso);

  /// construct from rank, eta, phi, isolation
  L1GctEmCand(unsigned rank, int phi, int eta, bool iso);

   /// destructor
 ~L1GctEmCand();
 
  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  unsigned rank() const { return m_data & 0x3f; }

  /// get eta bits
  int level1EtaIndex() const { return (m_data>>6) & 0x7; } // fix sign!

  /// get eta sign
  int level1EtaSign() const { return (m_data>>7) & 0x1; }

  /// get phi bits
  int level1PhiIndex() const { return (m_data>>10) & 0x1f; }

  /// which stream did this come from
  bool isolated() const { return m_iso; }

 private:

  uint16_t m_data;
  bool m_iso;

 };


std::ostream& operator<<(std::ostream& s, const L1GctEmCand& cand);



#endif 

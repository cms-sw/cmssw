#ifndef L1CALOCAND_H
#define L1CALOCAND_H

#include <boost/cstdint.hpp>
#include <ostream>

/*! \class L1CaloCand
 * \brief Base class for EM and jet candidates, for convenience (code re-use!)
 *
 */

/*! \author Jim Brooke
 *  \date June 2006
 */

class L1CaloCand
{
public:
  /// default constructor (required for vector initialisation etc.)
  L1CaloCand();

  /// construct from raw data
  L1CaloCand(uint16_t data);

  /// construct from separate rank, eta, phi
  L1CaloCand(int rank, int phi, int eta);

  /// destruct
  ~L1CaloCand();
  
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

#endif

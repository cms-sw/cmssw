#ifndef L1CALOEMCAND_H
#define L1CALOEMCAND_H

#include <boost/cstdint.hpp>
#include <ostream>


/*! \class L1CaloEmCand
 * \brief Level-1 Trigger EM candidate
 *
 */

/*! \author Jim Brooke
 *  \date June 2006
 */


class L1CaloEmCand {
public:

  /// default constructor (for vector initialisation etc.)
  L1CaloEmCand();

  /// construct from raw data
  L1CaloEmCand(uint16_t data);

  /// construct from rank, eta, phi, isolation
  L1CaloEmCand(int rank, int phi, int eta, bool iso);

   /// destructor
 ~L1CaloEmCand();
 
  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  int rank() const { return m_data & 0x3f; }
  
  /// get phi bits
  int phi() const { return (m_data>>6) & 0x1f; }
  
  /// get eta bits
  int eta() const { return (m_data>>11) & 0xf; }

  /// which stream did this come from
  bool isolated() const { return m_iso; }


 private:

  uint16_t m_data;

  bool m_iso;

 };


std::ostream& operator<<(std::ostream& s, const L1CaloEmCand& cand);



#endif 

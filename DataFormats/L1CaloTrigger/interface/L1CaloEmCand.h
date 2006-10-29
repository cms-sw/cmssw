#ifndef L1CALOEMCAND_H
#define L1CALOEMCAND_H

#include <boost/cstdint.hpp>
#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

/*! \class L1CaloEmCand
 * \brief Level-1 Region Calorimeter Trigger EM candidate
 *
 */

/*! \author Jim Brooke
 *  \date June 2006
 */


class L1CaloEmCand {
public:

  /// default constructor (for vector initialisation etc.)
  L1CaloEmCand();

  /// construct from raw data for unpacking
  L1CaloEmCand(uint16_t data, unsigned crate, bool iso);

  /// construct from components for emulation
  L1CaloEmCand(unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso);

   /// destructor
 ~L1CaloEmCand();
 
  /// get the raw data
  uint16_t raw() const { return m_data; }
  
  /// get rank bits
  unsigned rank() const { return m_data & 0x3f; }

  /// get RCT receiver card
  unsigned rctCard() const { return (m_data>>7) & 0x7; }

  /// get RCT region ID
  unsigned rctRegion() const { return (m_data>>6) & 0x1; }

  /// get RCT crate
  unsigned rctCrate() const { return m_rctCrate; }
  
  /// which stream did this come from
  bool isolated() const { return m_iso; }

  /// get DetID object
  L1CaloRegionDetId regionId() const { return L1CaloRegionDetId(false,rctCrate(),rctCard(),rctRegion()); }

 private:

  // rank, card and region ID are contained in the data on the cable
  uint16_t m_data;

  // members to store geographical information (crate/cable)
  // these should probably be packed into a single uint16_t (or m_data) ???
  unsigned m_rctCrate;
  bool m_iso;

 };


std::ostream& operator<<(std::ostream& s, const L1CaloEmCand& cand);



#endif 

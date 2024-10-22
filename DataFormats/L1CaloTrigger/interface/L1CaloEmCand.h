#ifndef L1CALOEMCAND_H
#define L1CALOEMCAND_H

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

  /// construct from raw data, no source - used by TextToDigi
  L1CaloEmCand(uint16_t data, unsigned crate, bool iso);

  /// construct from raw data with source - used by GCT unpacker
  /// last bool argument is a hack to distinguish this constructor from the next one!
  L1CaloEmCand(uint16_t data, unsigned crate, bool iso, uint16_t index, int16_t bx, bool dummy);

  /// construct from components for emulation
  L1CaloEmCand(unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso);

  /// construct from components for emulation (including index)
  L1CaloEmCand(unsigned rank, unsigned region, unsigned card, unsigned crate, bool iso, uint16_t index, int16_t bx);

  /// destructor
  ~L1CaloEmCand();

  /// get the raw data
  uint16_t raw() const { return m_data; }

  /// get rank bits
  unsigned rank() const { return m_data & 0x3f; }

  /// get RCT receiver card
  unsigned rctCard() const { return (m_data >> 7) & 0x7; }

  /// get RCT region ID
  unsigned rctRegion() const { return (m_data >> 6) & 0x1; }

  /// get RCT crate
  unsigned rctCrate() const { return m_rctCrate; }

  /// which stream did this come from
  bool isolated() const { return m_iso; }

  /// get index on cable
  unsigned index() const { return m_index; }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// get DetID object
  L1CaloRegionDetId regionId() const { return L1CaloRegionDetId(rctCrate(), rctCard(), rctRegion()); }

  /// set BX
  void setBx(int16_t bx);

  /// equality operator, including rank, isolation, position
  int operator==(const L1CaloEmCand& c) const {
    return ((m_data == c.raw() && m_iso == c.isolated() && m_rctCrate == c.rctCrate() &&
             this->regionId() == c.regionId()) ||
            (this->empty() && c.empty()));
  }

  /// inequality operator
  int operator!=(const L1CaloEmCand& c) const { return !(*this == c); }

  /// is there any information in the candidate
  bool empty() const { return (rank() == 0); }

private:
  // rank, card and region ID are contained in the data on the cable
  uint16_t m_data;

  // members to store geographical information (crate/cable)
  // these should probably be packed into a single uint16_t (or m_data) ???
  uint16_t m_rctCrate;
  bool m_iso;
  uint16_t m_index;
  int16_t m_bx;
};

std::ostream& operator<<(std::ostream& s, const L1CaloEmCand& cand);

#endif

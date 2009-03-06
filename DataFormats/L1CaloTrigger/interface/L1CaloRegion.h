#ifndef L1CALOREGION_H
#define L1CALOREGION_H

#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

/*!
 * \author Jim Brooke
 * \date May 2006
 */

/*!
 * \class L1CaloRegion
 * \brief A calorimeter trigger region (sum of 4x4 trigger towers)
 *
 * 
 *
 */


class L1CaloRegion
{
public:

  /// default constructor
  L1CaloRegion();

  /// constructor for RCT emulator (HB/HE regions)
  L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn);

  /// constructor for RCT emulator (HF regions)
  L1CaloRegion(unsigned et, bool fineGrain, unsigned crate, unsigned rgn);

  /// construct with GCT eta,phi indices, for testing GCT emulator - note argument ordering!
  L1CaloRegion(unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet, unsigned ieta, unsigned iphi);

  /// constructor from raw data and GCT indices for unpacking
  L1CaloRegion(uint16_t data, unsigned ieta, unsigned iphi, int16_t bx);

  /// destructor
  ~L1CaloRegion();
  

  // get/set methods for the data

  /// reset the data content (not position id!)
  void reset() { m_data = 0; }

  /// get raw data
  uint16_t raw() const { return m_data; }

  /// get Et
  unsigned et() const { return (isHf() ? m_data&0xff : m_data&0x3ff); }

  /// get overflow
  bool overFlow() const { return ((m_data>>10) & 0x1)!=0; }

  /// get tau veto bit
  bool tauVeto() const { return (isHf() ? false : fineGrain()); }

  /// get fine grain bit
  bool fineGrain() const { return ((m_data>>11) & 0x1)!=0; }

  /// get MIP bit
  bool mip() const { return ((m_data>>12) & 0x1)!=0; }

  /// get quiet bit
  bool quiet() const { return ((m_data>>13) & 0x1)!=0; }

  /// set BX (used in RCT emulator)
  void setBx(int16_t bx);

  /// set MIP bit (required for GCT emulator standalone operation)
  void setMip(bool mip);

  /// set quiet bit (required for GCT emulator standalone operation)
  void setQuiet(bool quiet);

  
  // get methods for the geographical information

  /// get global region ID
  L1CaloRegionDetId id() const { return m_id; }

  /// forward or central region
  bool isHf()   const { return m_id.isHf(); }
  bool isHbHe() const { return !m_id.isHf(); }

  /// get RCT crate ID
  unsigned rctCrate() const { return m_id.rctCrate(); }

  /// get RCT reciever card ID (valid output for HB/HE)
  unsigned rctCard() const { return m_id.rctCard(); }

  /// get RCT region index
  unsigned rctRegionIndex() const { return m_id.rctRegion(); }

  /// get local eta index (within RCT crate)
  unsigned rctEta() const { return m_id.rctEta(); }

  /// get local phi index (within RCT crate)
  unsigned rctPhi() const { return m_id.rctPhi(); } 

  /// get GCT eta index
  unsigned gctEta() const { return m_id.ieta(); }

  /// get GCT phi index
  unsigned gctPhi() const { return m_id.iphi(); }

  /// get bunch-crossing index
  int16_t bx() const { return m_bx; }

  /// equality operator, including rank, feature bits, and position
  int operator==(const L1CaloRegion& c) const { return ((m_data==c.raw() && m_id==c.id()) || (this->empty() && c.empty())); }

  /// inequality operator
  int operator!=(const L1CaloRegion& c) const { return !(*this == c); }

  /// is there any information in the candidate
  bool empty() const { return (m_data == 0); }

  /// print to stream
  friend std::ostream& operator << (std::ostream& os, const L1CaloRegion& reg);

 private:

  /// pack the raw data from arguments (used in constructors)
  void pack(unsigned et, bool overFlow, bool fineGrain, bool mip, bool quiet);

  /// region id
  L1CaloRegionDetId m_id;

  /// region data : et, overflow, fine grain/tau veto, mip and quiet bits
  uint16_t m_data;
  int16_t m_bx;

};


#endif /*L1CALOREGION_H*/

#ifndef L1CALOREGION_H
#define L1CALOREGION_H

#include <boost/cstdint.hpp>
#include <ostream>

/*!
 * \author Jim Brooke
 * \date May 2006
 */

/*!
 * \class L1CaloRegion
 * \brief A calorimeter trigger region (sum of 4x4 trigger towers)
 *
 *  Note that geographical information is not currently stored,
 *  awaiting advice on implementation.
 *
 *
 */


class L1CaloRegion
{
public:

  /// default constructor
  L1CaloRegion();

  /// constructor for emulation
  L1CaloRegion(unsigned et, bool overFlow, bool tauVeto, bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn);

  // constructor for unpacking
  L1CaloRegion(uint16_t data, unsigned crate, unsigned card, unsigned rgn);

  /// destructor
  ~L1CaloRegion();
  

  // get/set methods for the data

  /// get Et
  unsigned et() const { return (m_data & 0x3ff); }

  /// get overflow
  bool overFlow() const { return ((m_data>>10) & 0x1)!=0; }

  /// get tau veto bit
  bool tauVeto() const { return ((m_data>>11) & 0x1)!=0; }

  /// get MIP bit
  bool mip() const { return ((m_data>>12) & 0x1)!=0; }

  /// get quiet bit
  bool quiet() const { return ((m_data>>13) & 0x1)!=0; }

  /// set MIP bit (required for GCT emulator standalone operation)
  void setMip(bool mip);

  /// set quiet bit (required for GCT emulator standalone operation)
  void setQuiet(bool quiet);

  
  // get methods for the geographical information

  /// get RCT crate ID
  unsigned rctCrate() const ;

  /// get RCT reciever card ID (valid output for HB/HE)
  unsigned rctCard() const ;

  /// get RCT region index
  unsigned rctRegionIndex() const ;

  /// get local eta index (within RCT crate)
  unsigned rctEtaIndex() const ;

  /// get local phi index (within RCT crate)
  unsigned rctPhiIndex() const ; 

  /// get GCT source card ID
  unsigned gctCard() const ;

  /// get GCT eta index (global)
  unsigned gctEtaIndex() const ;

  /// get GCT phi index (global)
  unsigned gctPhiIndex() const ;

  /// get pseudorapidity
  float pseudorapidity() const ;

  /// get phi in radians
  float phi() const ;


  /// print to stream
  friend std::ostream& operator << (std::ostream& os, const L1CaloRegion& reg);

private:

  /// region data : et, overflow, tau veto, mip and quiet bits
  uint16_t m_data;

};


#endif /*L1GCTREGION_H_*/

#ifndef L1CALOMIPQUIETREGION_H
#define L1CALOMIPQUIETREGION_H

#include <ostream>

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

/*!
* \class L1CaloMipQuietRegion
* \brief Miniumum Ionising Particle (MIP) and Quiet bits for a calorimeter trigger region.
* 
* \author Robert Frazier
* $Revision: 1.2 $
* $Date: 2008/02/22 14:33:54 $
*/ 

class L1CaloMipQuietRegion
{
public:

  // ** Constructors/Destructors **

  /// Default constructor
  L1CaloMipQuietRegion();

  /// Constructor for RCT emulator (HB/HE regions)
  L1CaloMipQuietRegion(bool mip, bool quiet, unsigned crate, unsigned card, unsigned rgn, int16_t bx);

  /// Construct with GCT eta,phi indices, for testing GCT emulator
  L1CaloMipQuietRegion(bool mip, bool quiet, unsigned ieta, unsigned iphi, int16_t bx=0);

  /// Destructor
  ~L1CaloMipQuietRegion() {}
  
  
  // ** Operators **
  
  /// Equality operator; compares all data: MIP/Quiet bits, bunch crossing & geographical.
  bool operator==(const L1CaloMipQuietRegion& rhs) const;

  /// Inequality operator.
  bool operator!=(const L1CaloMipQuietRegion& rhs) const { return !(*this == rhs); }


  // ** Get methods for the data **

  uint8_t raw() const { return m_data; }  ///< Get raw data.

  bool mip() const { return (m_data & 0x1)!=0; }  ///< Get MIP bit.

  bool quiet() const { return ((m_data>>1) & 0x1)!=0; }  ///< Get Quiet bit.
  
  int16_t bx() const { return m_bx; }  ///< Get bunch crossing.


  // ** Set methods for the data **

  void setMip(bool mip) { mip ? m_data|=1 : m_data&=~1; }  ///< Set MIP bit.

  void setQuiet(bool quiet) { quiet ? m_data|=2 : m_data&=~2; }  ///< Set Quiet bit.

  void setBx(int16_t bx) { m_bx = bx; }  ///< Set bunch crossing. 


  // ** Get methods for geographical information **

  L1CaloRegionDetId id() const { return m_id; }  ///< Get global region ID.

  unsigned rctCrate() const { return m_id.rctCrate(); }  ///< Get RCT crate ID.

  unsigned rctCard() const { return m_id.rctCard(); }  ///< Get RCT reciever card ID.

  unsigned rctRegionIndex() const { return m_id.rctRegion(); }  ///< Get RCT region index.

  unsigned rctEta() const { return m_id.rctEta(); }  ///< Get local eta index (within RCT crate).

  unsigned rctPhi() const { return m_id.rctPhi(); }  ///< Get local phi index (within RCT crate). 

  unsigned gctEta() const { return m_id.ieta(); }  ///< Get GCT eta index.

  unsigned gctPhi() const { return m_id.iphi(); }  ///< Get GCT phi index.


  // ** Misc **

  /// Is the object empty? Currently always returns false.
  bool empty() const { return false; }

  /// Resets the data content - i.e. resets MIP/Quiet and bx, but not position ID!
  void reset() { m_data = 0; m_bx = 0; }


private:

  // ** Private Data **
  
  L1CaloRegionDetId m_id;  ///< Geographical info: region ID.
  
  uint8_t m_data;  ///< MIP and Quiet bits for the region, packed in bit0 + bit1 respectively.
  
  int16_t m_bx;  ///< Bunch crossing.


  // ** Private Methods **
  
  /// For use in constructors - packs MIP/Quiet bools up into m_data;
  void pack(bool mip, bool quiet) { m_data = (mip?1:0)|(quiet?2:0); }

};  

/// Stream insertion operator - no need to be a friend.
std::ostream& operator<< (std::ostream& os, const L1CaloMipQuietRegion& rhs);



#endif /*L1CALOMIPQUIETREGION_H*/

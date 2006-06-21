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
 */


class L1CaloRegion
{
public:

  /// constructor
  L1CaloRegion(unsigned id=0, unsigned et=0, bool overFlow=false, bool tauVeto=false, bool mip=false, bool quiet=false);

  /// destructor
  ~L1CaloRegion();
  
  /// get eta index (0-21) of the region
  unsigned etaIndex() const;

  /// get phi index (0-17) of the region
  unsigned phiIndex() const;

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


  /// set MIP bit (required because MIP/quiet bits arrive at different source card from the rest of the region!)
  void setMip(bool mip);

  /// set quiet bit (required because MIP/quiet bits arrive at different source card from the rest of the region!)
  void setQuiet(bool quiet);



  /// print to stream
  friend std::ostream& operator << (std::ostream& os, const L1CaloRegion& reg);

private:

  uint16_t m_id;

  /// region data : et, overflow, tau veto, mip and quiet bits
  uint16_t m_data;

};


#endif /*L1GCTREGION_H_*/

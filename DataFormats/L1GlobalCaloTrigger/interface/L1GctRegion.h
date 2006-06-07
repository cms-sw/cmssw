#ifndef L1GCTREGION_H_
#define L1GCTREGION_H_

#include <boost/cstdint.hpp>
#include <ostream>

/*
 * A calorimeter trigger region
 * author: Jim Brooke, Robert Frazier
 * date: 20/2/2006
 * 
 */

class L1GctRegion
{
public:
  L1GctRegion(unsigned eta=0, unsigned phi=0, unsigned et=0, bool overFlow=false, bool tauVeto=false, bool mip=false, bool quiet=false);
  ~L1GctRegion();
  
  // region position //

  /// get Region unique ID
  int id() const { return m_id; }
  
  /// get eta index (0-21) of the region
  int eta() const { return m_eta; }

  /// get phi index (0-17) of the region
  int phi() const { return m_phi; }

  /// get RCT crate index (0-17)
  int rctCrate() const { return L1GctMap::getMap()->rctCrate(*this); }

  // get region data //

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


  // set methods //

  /// set MIP bit (required because MIP/quiet bits arrive at different source card from the rest of the region!)
  void setMip(bool mip);

  /// set quiet bit (required because MIP/quiet bits arrive at different source card from the rest of the region!)
  void setQuiet(bool quiet);



  // print to stream
  friend std::ostream& operator << (std::ostream& os, const L1GctRegion& reg);

private:

  // region ID
  unsigned m_id;
  
  // position indices
  unsigned m_eta;
  unsigned m_phi;

  // region data : et, overflow, tau veto, mip and quiet bits
  uint16_t m_data;

};


#endif /*L1GCTREGION_H_*/

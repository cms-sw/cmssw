#ifndef L1GCTEMCAND_H_
#define L1GCTEMCAND_H_

#include <bitset>

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

using std::bitset;

/*! \file L1GctEmCand.h
 * \Header file for the Gct electron 
 *  candidate
 * 
 * \author: Jim Brooke
 *
 * Set methods added by Maria Hansen
 * \date: 15/03/2006
 */

typedef unsigned long int ULong;

class L1GctEmCand
{
public:
  L1GctEmCand(ULong data=0);
  L1GctEmCand(ULong rank, ULong eta, ULong phi);
  ~L1GctEmCand();
	
  /// set rank bits
  void setRank(unsigned rank) { m_rank = rank; }
  
  /// set eta bits
  void setEta(unsigned eta) { m_eta = eta; }
  
  /// set phi bits
  void setPhi(unsigned phi) { m_phi = phi; }
  
  
  /// get rank bits
  inline unsigned rank() const { return m_rank & 0x3f; }
  
  /// get eta bits
  inline unsigned eta() const { return m_eta & 0xf; }
  
  /// get phi bits
  inline unsigned phi() const { return m_phi & 0x1f; }

  /// convert to iso em digi
  L1GctIsoEm makeIsoEm();

  /// convert to non iso em digi
  L1GctNonIsoEm makeNonIsoEm();

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctEmCand& cand);

private:

    //DECLARE STATICS
    static const int RANK_BITWIDTH;
    static const int ETA_BITWIDTH;
    static const int PHI_BITWIDTH;

    unsigned m_rank;
    unsigned m_eta;
    unsigned m_phi;
      
};


#endif /*L1GCTEMCAND_H_*/

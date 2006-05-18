#ifndef L1GCTEMCAND_H_
#define L1GCTEMCAND_H_

#include <bitset>

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

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctEmCand& cand);
	
  ///
  /// set rank bits
  void setRank(unsigned long rank) { myRank = rank; }
  ///
  /// set eta bits
  void setEta(unsigned long eta) { myEta = eta; }
  ///
  /// set phi bits
  void setPhi(unsigned long phi) { myPhi = phi; }
  
  ///
  /// get rank bits
  inline unsigned long rank() const { return myRank.to_ulong(); }
  ///
  /// get eta bits
  inline unsigned long eta() const { return myEta.to_ulong(); }
  ///
  /// get phi bits
  inline unsigned long phi() const { return myPhi.to_ulong(); }

  ///
  /// overloaded greater than operator for sorting
  //  bool operator< (L1GctEmCand& c);

private:

    static const int RANK_BITWIDTH = 6;
    static const int ETA_BITWIDTH = 5;
    static const int PHI_BITWIDTH = 5;

    bitset<RANK_BITWIDTH> myRank;
    bitset<ETA_BITWIDTH> myEta;
    bitset<PHI_BITWIDTH> myPhi;
      
};

std::ostream& operator << (std::ostream& os, const L1GctEmCand& cand);

#endif /*L1GCTEMCAND_H_*/

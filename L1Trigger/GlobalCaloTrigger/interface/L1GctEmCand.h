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
	L1GctEmCand(ULong rank=0, ULong eta=0, ULong phi=0);
	~L1GctEmCand();
	
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
	inline unsigned long getRank() { return myRank.to_ulong(); }
	///
	/// get eta bits
	inline unsigned long getEta() { return myEta.to_ulong(); }
	///
	/// get phi bits
	inline unsigned long getPhi() { return myPhi.to_ulong(); }

private:

	bitset<6> myRank;
	bitset<5> myEta;
	bitset<4> myPhi;
      
};

#endif /*L1GCTEMCAND_H_*/

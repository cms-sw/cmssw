#ifndef L1GCTEMCAND_H_
#define L1GCTEMCAND_H_

#include <bitset>

using std::bitset;

/*! \file L1GctElectronSorter.h
 * \Header file for the Gct electron 
 *  candidate sorter class
 * 
 * \author: Jim Brook
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
	
	// set internal data
	void setRank(unsigned long rank) { myRank = rank; }
	void setEta(unsigned long eta) { myEta = eta; }
	void setPhi(unsigned long phi) { myPhi = phi; }

	// get internal data
	inline unsigned long getRank() { return myRank.to_ulong(); }
	inline unsigned long getEta() { return myEta.to_ulong(); }
	inline unsigned long getPhi() { return myPhi.to_ulong(); }

private:

	bitset<6> myRank;
	bitset<5> myEta;
	bitset<4> myPhi;
      
};

#endif /*L1GCTEMCAND_H_*/

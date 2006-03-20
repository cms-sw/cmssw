#ifndef L1GCTEMCAND_H_
#define L1GCTEMCAND_H_

#include <bitset>

using namespace std;
/*! \file L1GctElectronSorter.h
 * \Header file for the Gct electron 
 *  candidate sorter class
 * 
 * \author: Jim Brook
 *
 * Set methods added by Maria Hansen
 * \date: 15/03/2006
 */

class L1GctEmCand
{
public:
	L1GctEmCand();
	virtual ~L1GctEmCand();
	
	//need these the access the private variables rank,eta and phi
	
	void setRank(unsigned long inRank) {rank = inRank; }
	void setEta(unsigned long inEta) {eta = inEta; }
	void setPhi(unsigned long inPhi) {phi = inPhi; }
	inline unsigned long getRank() { return rank.to_ulong(); }
	inline unsigned long getEta() { return eta.to_ulong(); }
	inline unsigned long getPhi() { return phi.to_ulong(); }

private:

	bitset<6> rank;
	bitset<5> eta;
	bitset<4> phi;
      
};

#endif /*L1GCTEMCAND_H_*/

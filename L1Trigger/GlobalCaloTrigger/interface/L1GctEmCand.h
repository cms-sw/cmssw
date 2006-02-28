#ifndef L1GCTEMCAND_H_
#define L1GCTEMCAND_H_

#include <bitset>

using namespace std;

/*
 * Represents a GCT EM Candidate
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctEmCand
{
public:
	L1GctEmCand();
	virtual ~L1GctEmCand();
	
	inline unsigned long getRank() { return rank.to_ulong(); }
	inline unsigned long getEta() { return eta.to_ulong(); }
	inline unsigned long getPhi() { return phi.to_ulong(); }

private:

	bitset<6> rank;
	bitset<5> eta;
	bitset<4> phi;
	
};

#endif /*L1GCTEMCAND_H_*/

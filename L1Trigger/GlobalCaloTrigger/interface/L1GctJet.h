#ifndef L1GCTJET_H_
#define L1GCTJET_H_

#include <bitset>

using namespace std;

/*
 * A GCT jet candidate
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJet
{
public:
	L1GctJet();
	virtual ~L1GctJet();
	
	inline unsigned long getRank() { return rank.to_ulong(); }
	inline unsigned long getEta() { return eta.to_ulong(); }
	inline unsigned long getPhi() { return phi.to_ulong(); }

private:

	bitset<6> rank;
	bitset<5> eta;
	bitset<4> phi;	
	
};

#endif /*L1GCTJET_H_*/

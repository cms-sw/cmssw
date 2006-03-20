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

typedef unsigned long ULong;

class L1GctEmCand
{
public:
	L1GctEmCand(ULong rank=0, ULong eta=0, ULong phi=0);
//	L1GctJet(float et=0.; float eta=0.; float phi=0.);
	~L1GctEmCand();

	// set rank and position bits
	void setRank(ULong rank);
	void setEta(ULong eta);
	void setPhi(ULong phi);
	
	// set et and position, physically meaningful
	void setEt(float et);
	void setEta(float eta);
	void setPhi(float phi);
	
	// get rank and position bits
	inline ULong getRank() { return rank.to_ulong(); }
	inline ULong getEta() { return eta.to_ulong(); }
	inline ULong getPhi() { return phi.to_ulong(); }
	
	// get rank and position, physically meaningful 
	float getEt();
	float getEta();
	float getPhi();

	ostream& operator << (ostream& os, const L1GctEmCand& s);

private:

	bitset<6> rank;
	bitset<5> eta;
	bitset<4> phi;
	
};

#endif /*L1GCTEMCAND_H_*/

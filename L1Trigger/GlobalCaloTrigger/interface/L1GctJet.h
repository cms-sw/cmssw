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


typedef unsigned long int ULong;

class L1GctJet
{

public:

	L1GctJet(ULong rank=0, ULong eta=0, ULong phi=0);
//	L1GctJet(float et=0.; float eta=0.; float phi=0.);
	~L1GctJet();

	// set rank and position bits
	inline void setRank(ULong rank) { myRank = rank; }
	inline void setEta(ULong eta) { myEta = eta; }
	inline void setPhi(ULong phi) { myPhi = phi; }

	// set et and position, physically meaningful
	void setEt(float et);
	void setEta(float eta);
	void setPhi(float phi);
	
	// get rank and position bits
	inline ULong getRank() { return myRank.to_ulong(); }
	inline ULong getEta() { return myEta.to_ulong(); }
	inline ULong getPhi() { return myPhi.to_ulong(); }
	
	// get rank and position, physically meaningful 

	ostream& operator << (ostream& os, const L1GctJet& s);

private:

	bitset<6> myRank;
	bitset<5> myEta;
	bitset<4> myPhi;	
		
};

#endif /*L1GCTJET_H_*/

#ifndef L1GCTJET_H_
#define L1GCTJET_H_

#include <bitset>

using std::bitset;

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

	L1GctJet(ULong rank=0, ULong eta=0, ULong phi=0, bool tauVeto=true);
//	L1GctJet(float et=0.; float eta=0.; float phi=0.);
	~L1GctJet();

	// set rank and position bits
	inline void setRank(ULong rank) { myRank = rank; }
	inline void setEta(ULong eta) { myEta = eta; }
	inline void setPhi(ULong phi) { myPhi = phi; }
    inline void setTauVeto(bool tauVeto) { myTauVeto = tauVeto; }

	// get rank and position bits
	inline ULong getRank()const { return myRank.to_ulong(); }
	inline ULong getEta()const { return myEta.to_ulong(); }
	inline ULong getPhi()const { return myPhi.to_ulong(); }
    inline bool getTauVeto()const { return myTauVeto; }
	
	//ostream& operator << (ostream& os, const L1GctJet& s);

private:

	bitset<6> myRank;
	bitset<4> myEta;
	bitset<5> myPhi;
    bool myTauVeto;
		
};

#endif /*L1GCTJET_H_*/

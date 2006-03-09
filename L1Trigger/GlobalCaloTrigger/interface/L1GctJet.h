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

//typedefs for ease of use
typedef bitset<4> FourBit;
typedef bitset<5> FiveBit;
typedef bitset<6> SixBit;
typedef unsigned long int ULong;

class L1GctJet
{
public:
    L1GctJet();
	L1GctJet(ULong rank, ULong eta, ULong phi);
	virtual ~L1GctJet();
	
    // Getters
	ULong getRank() const { return m_rank.to_ulong(); }
	ULong getEta() const  { return m_eta.to_ulong(); }
	ULong getPhi() const { return m_phi.to_ulong(); }

    // Setters
    void setRank(ULong rank) { SixBit tempRank(rank); m_rank = tempRank; }
    void setEta(ULong eta) { FiveBit tempEta(eta); m_eta = tempEta; }
    void setPhi(ULong phi) { FourBit tempPhi(phi); m_phi = tempPhi; }

private:

	SixBit m_rank;
	FiveBit m_eta;
	FourBit m_phi;	
	
};

#endif /*L1GCTJET_H_*/

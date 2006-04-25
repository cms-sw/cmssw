#ifndef L1GCTREGION_H_
#define L1GCTREGION_H_

#include <bitset>

using namespace std;

/*
 * A calorimeter trigger region
 * as represented in the GCT
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */


typedef unsigned long int ULong;

class L1GctRegion
{
public:

	L1GctRegion(ULong et=0, bool mip=false, bool quiet=false, bool tauVeto=true, bool overFlow=false);
	~L1GctRegion();
	
    // Getters
	ULong getEt() const { return myEt.to_ulong(); }
	bool getMip() const { return myMip; }
	bool getQuiet() const { return myQuiet; }
    bool getTauVeto() const { return myTauVeto; }
    bool getOverFlow() const { return myOverFlow; }
    
    // Setters
    void setEt(ULong et) { myEt = et; } 
    void setMip(bool mip) { myMip = mip; }
    void setQuiet(bool quiet) { myQuiet = quiet; }
    void setTauVeto(bool tauVeto) { myTauVeto = tauVeto; }
    void setOverFlow(bool overFlow) { myOverFlow = overFlow; }

    //	ostream& operator << (ostream& os, const L1GctRegion& s);
    
    static const int ET_BITWIDTH = 10;
		
private:

	bitset<ET_BITWIDTH> myEt;
	bool myMip;
	bool myQuiet;
    bool myTauVeto;
    bool myOverFlow;
	
	
	
};

#endif /*L1GCTREGION_H_*/

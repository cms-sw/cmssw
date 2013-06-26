#ifndef L1GCTREGION_H_
#define L1GCTREGION_H_

#include <bitset>

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

	L1GctRegion(ULong et=0, bool mip=false, bool quiet=false);
	~L1GctRegion();
	
    // Getters
	ULong getEt() const { return myEt.to_ulong(); }
	bool getMip() const { return myMip; }
	bool getQuiet() const { return myQuiet; }
    
    // Setters
    void setEt(ULong et) { myEt = et; } 
    void setMip(bool mip) { myMip = mip; }
    void setQuiet(bool quiet) { myQuiet = quiet; }

	ostream& operator << (ostream& os, const L1GctRegion& s);
		
private:

	std::bitset<10> myEt;
	bool myMip;
	bool myQuiet;
	
};

#endif /*L1GCTREGION_H_*/

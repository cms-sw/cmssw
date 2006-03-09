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

//typedefs for ease of use
typedef bitset<10> TenBit;
typedef unsigned long int ULong;

class L1GctRegion
{
public:
    L1GctRegion();
	L1GctRegion(ULong et, bool mip, bool quiet);
	virtual ~L1GctRegion();
	
    // Getters
	ULong getEt() const { return m_et.to_ulong(); }
	bool getMip() const { return m_mip; }
	bool getQuiet() const { return m_quiet; }
    
    // Setters
    void setEt(ULong et) { TenBit tempEt(et); m_et = tempEt; } 
    void setMip(bool mip) { m_mip = mip; }
    void setQuiet(bool quiet) { m_quiet = quiet; }
	
private:

	TenBit m_et;
	bool m_mip;
	bool m_quiet;
	
};

#endif /*L1GCTREGION_H_*/

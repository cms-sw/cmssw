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

class L1GctRegion
{
public:
	L1GctRegion();
	virtual ~L1GctRegion();
	
	inline unsigned long getEt() { return et.to_ulong(); }
	inline bool getMip() { return mip; }
	inline bool getQuiet() { return quiet; }
	
private:

	bitset<10> et;
	bool mip;
	bool quiet;
	
};

#endif /*L1GCTREGION_H_*/

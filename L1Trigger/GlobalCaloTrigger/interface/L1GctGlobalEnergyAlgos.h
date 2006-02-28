#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include <bitset>

using namespace std;

/*
 * Emulates the GCT global energy algorithms
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctGlobalEnergyAlgos
{
public:
	L1GctGlobalEnergyAlgos();
	virtual ~L1GctGlobalEnergyAlgos();
	
	void process();
	
	inline unsigned long getEtMiss() { return etMiss.to_ulong(); }
	inline unsigned long getEtMissPhi()  { return etMissPhi.to_ulong(); }
	inline unsigned long getEtSum() { return etSum.to_ulong(); }
	inline unsigned long getEtHad() { return etHad.to_ulong(); }
	
private:
	
	bitset<13> etMiss;
	bitset<6> etMissPhi;
	bitset<13> etSum;
	bitset<13> etHad;

};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

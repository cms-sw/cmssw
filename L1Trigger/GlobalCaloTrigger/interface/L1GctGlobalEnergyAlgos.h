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
	~L1GctGlobalEnergyAlgos();

	// clear internal data
	void reset();
	
	// process the event
	void process();

	// set input data	
	void setInputSums(unsigned ex, unsigned ey);

	// return input data
	inline unsigned long getInputEx() { return inputEx.to_ulong();; }
	inline unsigned long getInputEy() { return inputEy.to_ulong();; }

	// return output data	
	inline unsigned long getEtMiss() { return outputEtMiss.to_ulong(); }
	inline unsigned long getEtMissPhi()  { return outputEtMissPhi.to_ulong(); }
	inline unsigned long getEtSum() { return outputEtSum.to_ulong(); }
	inline unsigned long getEtHad() { return outputEtHad.to_ulong(); }
	
private:
	
	// input data - need to confirm number of bits!
	bitset<12> inputEx;
	bitset<12> inputEy;
	
	// output data
	bitset<13> outputEtMiss;
	bitset<6> outputEtMissPhi;
	bitset<13> outputEtSum;
	bitset<13> outputEtHad;

};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

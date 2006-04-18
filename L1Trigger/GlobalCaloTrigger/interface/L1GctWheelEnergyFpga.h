#ifndef L1GCTWHEELENERGYFPGA_H_
#define L1GCTWHEELENERGYFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <bitset>
#include <vector>

using std::bitset;
using std::vector;

class L1GctJetLeafCard;

class L1GctWheelEnergyFpga : public L1GctProcessor
{
public:
	L1GctWheelEnergyFpga();
	L1GctWheelEnergyFpga(vector<L1GctJetLeafCard*> src);
	~L1GctWheelEnergyFpga();
	///
	/// clear internal buffers
	virtual void reset();
	///
	/// get input data from sources
	virtual void fetchInput();
	///
	/// process the data, fill output buffers
	virtual void process();
	///
	/// set input data
	void setInputEnergy(int i, unsigned ex, unsigned ey);
	
	// get input data
	inline unsigned long getInputEx() { return inputEx.to_ulong(); }
	inline unsigned long getInputEy() { return inputEy.to_ulong(); }
	
	// get output data
	inline unsigned long getOutputEx() { return outputEx.to_ulong(); }
	inline unsigned long getOutputEy() { return outputEy.to_ulong(); }
	inline unsigned long getOutputHt() { return outputHt.to_ulong(); }

private:

	///
	/// the jet leaf card
	vector<L1GctJetLeafCard*> inputLeafCards;

	// input data - need to confirm number of bits!
	bitset<12> inputEx;
	bitset<12> inputEy;
	
	// output data
	bitset<13> outputEx;
	bitset<13> outputEy;
	bitset<13> outputHt;
	
	
};

#endif /*L1GCTWHEELENERGYFPGA_H_*/

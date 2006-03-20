#ifndef L1GCTWHEELENERGYFPGA_H_
#define L1GCTWHEELENERGYFPGA_H_

class L1GctWheelEnergyFpga
{
public:
	L1GctWheelEnergyFpga();
	~L1GctWheelEnergyFpga();
	
	// clear internal data
	void reset();
	
	// process the event
	void process();

	// set input data
	void setInputSums(unsigned ex, unsigned ey);
	
	// get input data
	inline unsigned long getInputEx() { return inputEx; }
	inline unsigned long getInputEy() { return inputEy; }
	
	// get output data
	inline unsigned long getOutputEx() { return outputEx; }
	inline unsigned long getOutputEy() { return outputEy; }
	inline unsigned long getOutputHt() { return outputHt; }

private:

	// input data - need to confirm number of bits!
	bitset<12> inputEx;
	bitset<12> inputEy;
	
	// output data
	bitset<13> outputEx;
	bitset<13> outputEy;
	bitset<13> outputEtHad;
	
	
};

#endif /*L1GCTWHEELENERGYFPGA_H_*/

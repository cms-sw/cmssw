#ifndef L1GCTWHEELJETFPGA_H_
#define L1GCTWHEELJETFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

class L1GctWheelJetFpga
{
public:
	L1GctWheelJetFpga();
	~L1GctWheelJetFpga();

	// clear internal data
	void reset();
	
	// perform processing and fill output array
	void process();

	// set the i-th input jet
	void setInputJet(int i, L1GctJet jet); 
	
	// get the input jets
	inline vector<L1GctJet> getInputJets() { return inputJets; }
	
	// get the output jets
	inline vector<L1GctJet> getOutputJets() { return outputJets; }
	
	
private:

	// input data
	// this should be a fixed size array!
	// with meaning assigned to the positions
	vector<L1GctJet> inputJets;

	// output data
	// this should be a fixed size array
	// perhaps with meaning assigned to the positions
	// (eg. central 0-3, forward 4-7, tau 8-11)
	vector<L1GctJet> outputJets;

};

#endif /*L1GCTWHEELJETFPGA_H_*/

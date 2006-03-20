#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <vector>

class L1GctElectronFinalSort
{
public:
	L1GctElectronFinalSort();
	~L1GctElectronFinalSort();

	// clear internal data
	void reset();
	
	// process the event
	void process();

	// return input data
	void setInputEmCand(int i, L1GctEmCand cand);
	
	// return output data
	inline vector<L1GctEmCand> getInputCands() { return inputCands; }
	inline vector<L1GctEmCand> getOutputCands() { return outputCands; }

private:

	// input data
	vector<L1GctEmCand> inputCands;
	
	// output data
	vector<L1GctEmCand> outputCands;

};

#endif /*L1GCTELECTRONFINALSORT_H_*/

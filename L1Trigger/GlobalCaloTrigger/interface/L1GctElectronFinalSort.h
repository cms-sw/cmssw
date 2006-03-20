#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

class L1GctElectronFinalSort
{
public:
	L1GctElectronFinalSort();
	~L1GctElectronFinalSort();
	
	void setInputEmCand(int i, L1GctEmCand cand);
	void process();
	
	inline vector<L1GctEmCand> getInput() { return inputCands; }
	inline vector<L1GctEmCand> getOutput() { return outputCands; }

private:

	// input data
	vector<L1GctEmCand> inputCands;
	
	// output data
	vector<L1GctEmCand> outputCands;

};

#endif /*L1GCTELECTRONFINALSORT_H_*/

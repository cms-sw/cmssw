#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include <vector>

using std::vector;

class L1GctElectronFinalSort
{
public:
	L1GctElectronFinalSort();
	~L1GctElectronFinalSort();

	// clear internal data
	void reset();

	//set inputs
	void setSortedInput(L1GctEmCand cand);

	// process the event
	void process();

	// return output data
	inline vector<L1GctEmCand> getInputCands() { return inputCands; }
	inline vector<L1GctEmCand> getOutputCands() { return outputCands; }

private:
	
	//Already sorted input objects
	L1GctElectronSorter sortedCands;

	// input data
	vector<L1GctEmCand> inputCands;

	// output data
	vector<L1GctEmCand> outputCands;
};

#endif /*L1GCTELECTRONFINALSORT_H_*/

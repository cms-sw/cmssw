#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include <vector>

using std::vector;

class L1GctElectronSorter;

class L1GctElectronFinalSort : public L1GctProcessor
{
public:
	L1GctElectronFinalSort();
	L1GctElectronFinalSort(vector<L1GctElectronSorter*> src);
	~L1GctElectronFinalSort();
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
	/// return input data
	void setInputEmCand(int i, L1GctEmCand cand);

	//set inputs
	void setSortedInput(L1GctEmCand cand);

	// return input data
	inline vector<L1GctEmCand> getInputCands() { return inputCands; }
	///
	/// return output data
	inline vector<L1GctEmCand> getOutputCands() { return outputCands; }

private:
	
	//Already sorted input objects
	L1GctElectronSorter sortedCands;

	///
	/// the 1st stage electron sorters
	vector<L1GctElectronSorter*> theEmSorters;
	///
	/// input data
	vector<L1GctEmCand> inputCands;
	///
	/// output data

	vector<L1GctEmCand> outputCands;
};

#endif /*L1GCTELECTRONFINALSORT_H_*/

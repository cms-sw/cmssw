#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>
#include <functional>

using std::vector;
using std::binary_function;

class L1GctEmLeafCard;

class L1GctElectronFinalSort : public L1GctProcessor
{
public:
	L1GctElectronFinalSort(bool iso);
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
	/// set input sources
	void setInputLeafCard(int i, L1GctEmLeafCard* card);
	///
	/// set input data
	void setInputEmCand(int i, L1GctEmCand cand);
	///
	/// return input data
	inline vector<L1GctEmCand> getInputCands() { return inputCands; }
	///
	/// return output data
	inline vector<L1GctEmCand> getOutputCands() { return outputCands; }

 private:

	// comparison operator for sort
	struct rank_gt : public binary_function<L1GctEmCand, L1GctEmCand, bool> {
	  bool operator()(const L1GctEmCand& x, const L1GctEmCand& y) { return x.rank() > y.rank(); }
	};

 private:

	///
	/// type of em cand
	bool getIsoEmCands;
	///
	/// the 1st stage electron sorters
	vector<L1GctEmLeafCard*> theLeafCards;
	///
	/// input data
	vector<L1GctEmCand> inputCands;
	///
	/// output data
	vector<L1GctEmCand> outputCands;

};

#endif /*L1GCTELECTRONFINALSORT_H_*/

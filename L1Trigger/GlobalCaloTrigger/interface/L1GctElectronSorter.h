#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>

using std::vector;

class L1GctSourceCard;

///
/// Represents a GCT Electron Sort algorithm
/// author: Maria Hansen
/// date: 20/2/2006
/// 
///

class L1GctElectronSorter : public L1GctProcessor
{
public:
	L1GctElectronSorter();
	L1GctElectronSorter(vector<L1GctSourceCard*> src);	
	~L1GctElectronSorter();
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
	/// set input candidate
	void setInputEmCand(L1GctEmCand cand);
	///	
	/// get input candidates
	inline vector<L1GctEmCand> getInput() { return inputCands; }
	///
	/// get output candidates
	inline vector<L1GctEmCand> getOutput() { return outputCands; }
	
private:
	
	///
	/// source card input
	vector<L1GctSourceCard*> theSCs;
	///
	/// input data
	vector<L1GctEmCand> inputCands;
	///
	/// output data
	vector<L1GctEmCand> outputCands;
	
};

#endif /*L1GCTELECTRONSORTER_H_*/

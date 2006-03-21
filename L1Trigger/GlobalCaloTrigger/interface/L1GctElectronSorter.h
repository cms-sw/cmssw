#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_
//NOTE, to be changed...
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <vector>

using std::vector;

/*
 * Represents a GCT Electron Sort algorithm
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctElectronSorter
{
public:
	L1GctElectronSorter();
	virtual ~L1GctElectronSorter();

	void setInputEmCand(L1GctEmCand cand);
	void process();
	
	inline vector<L1GctEmCand> getInput() { return inputCands; }
	inline vector<L1GctEmCand> getOutput() { return outputCands; }
	
private:
	
	// input data
	vector<L1GctEmCand> inputCands;
	
	// output data
	vector<L1GctEmCand> outputCands;
	
};

#endif /*L1GCTELECTRONSORTER_H_*/

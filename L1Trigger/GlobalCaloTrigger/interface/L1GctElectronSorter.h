#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"

#include <vector>

using namespace std;

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
	~L1GctElectronSorter();

	// clear internal data
	void reset();
	
	// process the event
	void process();

	// set input data
	void setInputEmCand(int i, L1GctEmCand cand);
	
	// return input data
	inline vector<L1GctEmCand> getInputCands() { return inputCands; }

	// return output data
	inline vector<L1GctEmCand> getOutputCands() { return outputCands; }
	
private:
	
	// input data
	vector<L1GctEmCand> inputCands;
	
	// output data
	vector<L1GctEmCand> outputCands;
	
};

#endif /*L1GCTELECTRONSORTER_H_*/

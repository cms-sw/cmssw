#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "L1GctEmCand.h"

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
	virtual ~L1GctElectronSorter();

	void process();
	
	inline vector<L1GctEmCand> getOutput() { return outputCands; }
	
private:
	
	vector<L1GctEmCand> outputCands;
	
};

#endif /*L1GCTELECTRONSORTER_H_*/

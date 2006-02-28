#ifndef L1GCTELECTRONLEAFCARD_H_
#define L1GCTELECTRONLEAFCARD_H_

#include "L1GctElectronSorter.h"
#include "L1GctSourceCard.h"

#include <vector>

using namespace std;

/**
  * Represents a GCT Leaf Card
  * programmed to sort EM candidates
  * author: Jim Brooke
  * date: 20/2/2006
  * 
  **/

class L1GctElectronLeafCard {
public:
	L1GctElectronLeafCard();
	virtual ~L1GctElectronLeafCard();

	void addSource(L1GctSourceCard* card);

	void process();
	
	vector<L1GctEmCand> getInput();
	vector<L1GctEmCand> getOutput();
	
private:

	// processing
	L1GctElectronSorter finalSort;

	// pointers to data source
	vector<L1GctSourceCard*> sourceCards;
	
	// input data buffer
	vector<L1GctEmCand> inputCands;
	
};

#endif /*L1GCTELECTRONLEAFCARD_H_*/

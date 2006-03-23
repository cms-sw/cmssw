#ifndef L1GCTELECTRONLEAFCARD_H_
#define L1GCTELECTRONLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>

using namespace std;

///
/// Represents a GCT Leaf Card
/// programmed to sort EM candidates
/// author: Jim Brooke
/// date: 20/2/2006
/// 
///

class L1GctElectronLeafCard {
public:
	L1GctElectronLeafCard();
	~L1GctElectronLeafCard();

	///
	/// add a source card as input
	void addSource(L1GctSourceCard* card);

	///
	/// clear buffers
	void reset();

	///
	/// process the event
	void process();
	
	///
	/// get the input candidates
	vector<L1GctEmCand> getInput();

	///
	/// get the output candidates
	vector<L1GctEmCand> getOutput();
	
private:

	///
	/// processing
	L1GctElectronSorter finalSort;

	///
	/// pointers to data source
	vector<L1GctSourceCard*> sourceCards;
	
	///
	/// input data buffer
	vector<L1GctEmCand> inputCands;
	
};

#endif /*L1GCTELECTRONLEAFCARD_H_*/

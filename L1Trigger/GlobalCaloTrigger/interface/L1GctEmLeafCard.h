#ifndef L1GCTELECTRONLEAFCARD_H_
#define L1GCTELECTRONLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
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

class L1GctEmLeafCard : L1GctProcessor {
public:
	L1GctEmLeafCard();
	~L1GctEmLeafCard();
	///
	/// clear buffers
	virtual void reset();
	///
	/// fetch input data
	virtual void fetchInput();
	///
	/// process the event
	virtual void process();	
	///
	/// add a source card as input
	void setInputSourceCard(int i, L1GctSourceCard* sc);
	///
	/// get the output candidates
	vector<L1GctEmCand> getOutputIsoEmCands();
	///
	/// get the output candidates
	vector<L1GctEmCand> getOutputNonIsoEmCands();
	
private:

	///
	/// processing
	vector<L1GctElectronSorter*> m_sorters;
	///
	/// pointers to data source
	vector<L1GctSourceCard*> m_sourceCards;
      
};

#endif /*L1GCTELECTRONLEAFCARD_H_*/

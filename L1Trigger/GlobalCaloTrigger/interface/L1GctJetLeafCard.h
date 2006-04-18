#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>

using namespace std;

/*
 * Represents a GCT Leaf Card
 * programmed for jet finding
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetLeafCard
{
public:
	L1GctJetLeafCard();
	~L1GctJetLeafCard();

	void addSource(L1GctSourceCard* card);
	
	///
	/// clear the buffers
	void reset();
	///
	/// run the algorithms
	void process();
	
	///
	/// get the jet output
	vector<L1GctJet> theJetOutput();
	///
	/// get the energy outputs

private:

	// internal algorithms
	L1GctJetFinder* jetFinderA;
	L1GctJetFinder* jetFinderB;
	L1GctJetFinder* jetFinderC;	

	// pointers to data source
	vector<L1GctSourceCard*> sourceCards;

};

#endif /*L1GCTJETLEAFCARD_H_*/

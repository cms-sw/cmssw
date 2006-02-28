#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1GctJetFinder.h"
#include "L1GctSourceCard.h"

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
	virtual ~L1GctJetLeafCard();

	void addSource(L1GctSourceCard* card);
	
	void process();

private:

	// internal algorithms
	L1GctJetFinder* jetFinderA;
	L1GctJetFinder* jetFinderB;
	L1GctJetFinder* jetFinderC;	

	// pointers to data source
	vector<L1GctSourceCard*> sourceCards;

};

#endif /*L1GCTJETLEAFCARD_H_*/

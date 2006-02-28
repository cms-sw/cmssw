#ifndef L1GCTMUONLEAFCARD_H_
#define L1GCTMUONLEAFCARD_H_

#include "L1GctSourceCard.h"

#include <vector>

using namespace std;

/*
 * Represents a GCT Concentrator Card
 * programmed to forward muon data
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctMuonLeafCard
{
public:
	L1GctMuonLeafCard();
	virtual ~L1GctMuonLeafCard();

	void addSource(L1GctSourceCard* card);
		
	void process();

private:

	// pointers to data source
	vector<L1GctSourceCard*> sourceCards;
	
};

#endif /*L1GCTMUONLEAFCARD_H_*/

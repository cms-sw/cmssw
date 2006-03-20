#ifndef L1GCTMUONCONCENTRATORCARD_H_
#define L1GCTMUONCONCENTRATORCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctMuonLeafCard.h"

using namespace std;

/*
 * Represents a GCT Concentrator Card
 * programmed to process Muon data
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctMuonConcentratorCard
{
public:
	L1GctMuonConcentratorCard();
	~L1GctMuonConcentratorCard();
	
	void process();
	
	
private:

	L1GctMuonLeafCard* muonLeafCard;
	
};

#endif /*L1GCTMUONCONCENTRATORCARD_H_*/

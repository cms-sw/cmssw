#ifndef L1GCTSOURCECARD_H_
#define L1GCTSOURCECARD_H_

#include "L1GctEmCand.h"
#include "L1GctRegion.h"

#include <vector>
#include <bitset>

using namespace std;

class L1RctCrate;

/**
  * Represents a GCT Source Card
  * author: Jim Brooke
  * date: 20/2/2006
  * 
  **/

class L1GctSourceCard
{
public:
	L1GctSourceCard();
	virtual ~L1GctSourceCard();

	void setRctInputCrate(L1RctCrate* rc);

	void process();
	
	vector<L1GctEmCand> getIsoElectrons();
	vector<L1GctEmCand> getNonIsoElectrons();
	vector<L1GctRegion> getRegions();
	bitset<14> getMipBits();
	bitset<14> getQuietBits();

private:

	// need a pointer to the RCT crate...
	L1RctCrate* rctCrate;
	
};

#endif /*L1GCTSOURCECARD_H_*/

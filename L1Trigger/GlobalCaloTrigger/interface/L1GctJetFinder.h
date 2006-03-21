#ifndef L1GCTJETFINDER_H_
#define L1GCTJETFINDER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

#include <vector>
using namespace std;

/*
 * 
 * The GCT Jet finding algorithm
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetFinder
{
public:
	L1GctJetFinder();
	~L1GctJetFinder();

	// clear internal data
	void reset();
	
	// process the event
	void process();

	// set input data		
	void setInputRegion(int i, L1GctRegion rgn) {};

	// return input data	
	inline vector<L1GctRegion> getInputRegions() { return inputRegions; }

	// return output data
	inline vector<L1GctJet> getJets() { return outputJets; }
	inline unsigned long getHt() { return outputHt.to_ulong(); }
	
	// need method(s) to return jet counts - need to decide type!
		
private:

	// input data
	vector<L1GctRegion> inputRegions;

	// output jets
	vector<L1GctJet> outputJets;

	// output Ht - need to confirm number of bits
	bitset<12> outputHt;
	
	// jet count output - need to decide data type!
	//vector<bitset<4>> outputJetCounts;
	
};

#endif /*L1GCTJETFINDER_H_*/

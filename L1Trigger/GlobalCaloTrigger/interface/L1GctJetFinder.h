#ifndef L1GCTJETFINDER_H_
#define L1GCTJETFINDER_H_

#include "L1GctJet.h"

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
	virtual ~L1GctJetFinder();

	void setInputRegion(int i, L1GctRegion rgn);
	void process();
	
	inline vector<L1GctRegion> getInputRegions() { return inputRegions; }
	inline vector<L1GctJet> getJets() { return outputJets; }
	
	// more member functions needed for other outputs
		
private:

	// input data
	vector<L1GctRegion> inputRegions;

	// output data
	vector<L1GctJet> outputJets;
	
	// more data members needed for other outputs

};

#endif /*L1GCTJETFINDER_H_*/

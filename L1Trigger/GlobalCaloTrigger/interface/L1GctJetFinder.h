#ifndef L1GCTJETFINDER_H_
#define L1GCTJETFINDER_H_

#include "L1GctJet.h"

#include <vector>
using namespace std;

/*
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

	void process();

	inline vector<L1GctJet> getJets() { return outputJets; }
		
private:

	vector<L1GctJet> outputJets;

};

#endif /*L1GCTJETFINDER_H_*/

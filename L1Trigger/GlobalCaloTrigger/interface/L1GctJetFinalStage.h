#ifndef L1GCTJETFINALSTAGE_H_
#define L1GCTJETFINALSTAGE_H_

#include "L1GctJet.h"

#include <vector>

using namespace std;

/*
 * The GCT Jet classify and sort algorithms
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetFinalStage
{
public:
	L1GctJetFinalStage();
	virtual ~L1GctJetFinalStage();
	
	void setInputJet(int i, L1GctJet jet);
	void process();
	
	inline vector<L1GctJet> getCentralJets() { return centralJets; }
	inline vector<L1GctJet> getForwardJets() { return forwardJets; }
	inline vector<L1GctJet> getTauJets() { return tauJets; }

private:

	// input data
	vector<L1GctJet> inputJets;

	// output data
	vector<L1GctJet> centralJets;
	vector<L1GctJet> forwardJets;
	vector<L1GctJet> tauJets;
	
};

#endif /*L1GCTJETFINALSTAGE_H_*/

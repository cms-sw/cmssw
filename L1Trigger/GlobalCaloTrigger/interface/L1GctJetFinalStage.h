#ifndef L1GCTJETFINALSTAGE_H_
#define L1GCTJETFINALSTAGE_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>

using std::vector;

class L1GctWheelJetFpga;

/*
 * The GCT Jet classify and sort algorithms
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetFinalStage : public L1GctProcessor
{
public:
	L1GctJetFinalStage();
	L1GctJetFinalStage(vector<L1GctWheelJetFpga*> src);
	~L1GctJetFinalStage();
	///
	/// clear internal buffers
	virtual void reset();
	///
	/// get input data from sources
	virtual void fetchInput();
	///
	/// process the data, fill output buffers
	virtual void process();
	///	
	// set input data		
	void setInputJet(int i, L1GctJet jet);

	// return input data
	inline vector<L1GctJet> getInputJets() { return inputJets; }

	// return output data
	inline vector<L1GctJet> getCentralJets() { return centralJets; }
	inline vector<L1GctJet> getForwardJets() { return forwardJets; }
	inline vector<L1GctJet> getTauJets() { return tauJets; }

private:

	///
	/// wheel jet FPGAs
	vector<L1GctWheelJetFpga*> theWheelFpgas;
	
	// input data - need to confirm number of jets!
	vector<L1GctJet> inputJets;

	// output data
	vector<L1GctJet> centralJets;
	vector<L1GctJet> forwardJets;
	vector<L1GctJet> tauJets;
	
};

#endif /*L1GCTJETFINALSTAGE_H_*/

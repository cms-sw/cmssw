#ifndef L1GCTEVENT_H_
#define L1GCTEVENT_H_

/**
 * 
 * Container class for the GCT DAQ record
 * author : Jim Brooke
 * 
 **/

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include <vector>
#include <bitset>

using std::bitset;

class L1GctEvent
{
public:
	L1GctEvent();
	~L1GctEvent();
	
	// access to system input
	vector<L1GctEmCand> getInputIsoElectrons();
	vector<L1GctEmCand> getInputNonIsoElectrons();
	vector<L1GctRegion> getInputRegions();

	// access to system output
	vector<L1GctEmCand> getOutputIsoElectrons();
	vector<L1GctEmCand> getOutputNonIsoElectrons();
	
	vector<L1GctJet> getOutputCentralJets();
	vector<L1GctJet> getOutputForwardJets();
	vector<L1GctJet> getOutputTauJets();
	
	//ostream& operator << (ostream& os, const L1GctEvent& s);
		
private:

	// system input data
	vector<L1GctEmCand> inputIsoElectrons;
	vector<L1GctEmCand> inputNonIsoElectrons;
	vector<L1GctRegion> inputRegions;

	// system output data
	vector<L1GctEmCand> outputIsoElectrons;
	vector<L1GctEmCand> outputNonIsoElectrons;
	
	vector<L1GctJet> outputCentralJets;
	vector<L1GctJet> outputForwardJets;
	vector<L1GctJet> outputTauJets;

	bitset<13> etMiss;
	bitset<6> etMissPhi;
	bitset<13> etSum;
	bitset<13> etHad;
	
};

#endif /*L1GCTEVENT_H_*/

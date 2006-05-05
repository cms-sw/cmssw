#ifndef L1GCTWHEELJETFPGA_H_
#define L1GCTWHEELJETFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>

using std::vector;

class L1GctJetLeafCard;

class L1GctWheelJetFpga : public L1GctProcessor
{
public:
	L1GctWheelJetFpga(int id);
	~L1GctWheelJetFpga();
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
	/// set input data		
	void setInputJet(int i, L1GctJet jet); 
	
	// get the input jets
	inline vector<L1GctJet> getInputJets() { return inputJets; }
	
	// get the output jets
	inline vector<L1GctJet> getOutputJets() { return outputJets; }
	
	
private:
	///
	/// algo ID
	int m_id;
	///
	/// the jet leaf cards
	vector<L1GctJetLeafCard*> inputLeafCards;
	
	// input data
	// this should be a fixed size array!
	// with meaning assigned to the positions
	vector<L1GctJet> inputJets;

	// output data
	// this should be a fixed size array
	// perhaps with meaning assigned to the positions
	// (eg. central 0-3, forward 4-7, tau 8-11)
	vector<L1GctJet> outputJets;

};

#endif /*L1GCTWHEELJETFPGA_H_*/

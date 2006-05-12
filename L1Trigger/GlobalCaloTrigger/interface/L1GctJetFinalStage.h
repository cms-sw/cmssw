#ifndef L1GCTJETFINALSTAGE_H_
#define L1GCTJETFINALSTAGE_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>
#include <bitset>

using std::vector;
using std::bitset;

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
	~L1GctJetFinalStage();

	/// clear internal buffers
	virtual void reset();

	/// get input data from sources
	virtual void fetchInput();

	/// process the data, fill output buffers
	virtual void process();
    	
	// set input data		
	void setInputJet(int i, L1GctJetCand jet);

    /// set the wheel jet fpga pointers
    void setInputWheelJetFpga(int i, L1GctWheelJetFpga* wjf);

	// return input data
	std::vector<L1GctJetCand> getInputJets() const { return m_inputJets; }

	// return output data
	std::vector<L1GctJetCand> getCentralJets() const { return m_centralJets; }
	std::vector<L1GctJetCand> getForwardJets() const { return m_forwardJets; }
	std::vector<L1GctJetCand> getTauJets() const { return m_tauJets; }

	inline unsigned long getHtBoundaryJets()               { return outputHtBoundaryJets.to_ulong(); }
        inline unsigned long getJcBoundaryJets(unsigned jcnum) { return outputJcBoundaryJets[jcnum].to_ulong(); }
private:

	/// wheel jet FPGAs
	std::vector<L1GctWheelJetFpga*> m_wheelFpgas;
	
	// input data - need to confirm number of jets!
	std::vector<L1GctJetCand> m_inputJets;

	// output data
	std::vector<L1GctJetCand> m_centralJets;
	std::vector<L1GctJetCand> m_forwardJets;
	std::vector<L1GctJetCand> m_tauJets;
	
	// data sent to GlobalEnergyAlgos
        typedef bitset<3> JcBoundType;
	bitset<13> outputHtBoundaryJets;
        vector<JcBoundType> outputJcBoundaryJets;
};

#endif /*L1GCTJETFINALSTAGE_H_*/

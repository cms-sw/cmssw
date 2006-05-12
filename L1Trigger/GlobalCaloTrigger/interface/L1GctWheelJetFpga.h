#ifndef L1GCTWHEELJETFPGA_H_
#define L1GCTWHEELJETFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>
#include <bitset>

using std::vector;
using std::bitset;

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
	/// set input sources
	void setInputLeafCard(int i, L1GctJetLeafCard* card);
	///
	/// set input data		
	void setInputJet(int i, L1GctJet jet); 
	void setInputHt (int i, unsigned ht);
	
	// get the input jets
	inline vector<L1GctJet> getInputJets() { return inputJets; }
	
	// get the input Ht
	inline unsigned long getInputHt(unsigned leafnum) { return inputHt[leafnum].to_ulong(); }
	
	// get the output jets
	inline vector<L1GctJet> getOutputJets() { return outputJets; }
	
	// get the output Ht and jet counts
	inline unsigned long getOutputHt()               { return outputHt.to_ulong(); }
        inline unsigned long getOutputJc(unsigned jcnum) { return outputJc[jcnum].to_ulong(); }
	
private:
	///
	/// algo ID
	int m_id;
	///
	/// the jet leaf cards
	vector<L1GctJetLeafCard*> m_inputLeafCards;
	
	// input data
	// this should be a fixed size array!
	// with meaning assigned to the positions
	vector<L1GctJet> inputJets;

	// input Ht sums from each leaf card
	static const int NUM_BITS_ENERGY_DATA = 13;
	static const int OVERFLOW_BIT = NUM_BITS_ENERGY_DATA - 1;

        static const int Emax = (1<<NUM_BITS_ENERGY_DATA);
        static const int signedEmax = (Emax>>1);

	// input data - need to confirm number of bits!
        typedef bitset<NUM_BITS_ENERGY_DATA> InputEnergyType;
	vector<InputEnergyType> inputHt;


	// output data
	// this should be a fixed size array
	// perhaps with meaning assigned to the positions
	// (eg. central 0-3, forward 4-7, tau 8-11)
	vector<L1GctJet> outputJets;

	// data sent to GlobalEnergyAlgos
        typedef bitset<3> JcWheelType;
	bitset<NUM_BITS_ENERGY_DATA> outputHt;
        vector<JcWheelType> outputJc;
};

#endif /*L1GCTWHEELJETFPGA_H_*/

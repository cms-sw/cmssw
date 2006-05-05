#ifndef L1GCTWHEELENERGYFPGA_H_
#define L1GCTWHEELENERGYFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <bitset>
#include <vector>

using namespace std;

/* using std::bitset; */
/* using std::vector; */

class L1GctJetLeafCard;

class L1GctWheelEnergyFpga : public L1GctProcessor
{
public:
	L1GctWheelEnergyFpga(int id);
	~L1GctWheelEnergyFpga();
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
	/// assign data sources
	void setInputLeafCard (int i, L1GctJetLeafCard* leaf);
	///
	/// set input data
	void setInputEnergy(int i, int ex, int ey, unsigned et);
	
	// get input data
	inline unsigned long getInputEx(unsigned leafnum) { return inputEx[leafnum].to_ulong(); }
	inline unsigned long getInputEy(unsigned leafnum) { return inputEy[leafnum].to_ulong(); }
	inline unsigned long getInputEt(unsigned leafnum) { return inputEt[leafnum].to_ulong(); }
	
	// get output data
	inline unsigned long getOutputEx() { return outputEx.to_ulong(); }
	inline unsigned long getOutputEy() { return outputEy.to_ulong(); }
	inline unsigned long getOutputEt() { return outputEt.to_ulong(); }

private:

	///
	/// algo ID
	int m_id;
	///
	/// the jet leaf card
	vector<L1GctJetLeafCard*> m_inputLeafCards;

	static const int NUM_BITS_ENERGY_DATA = 13;
	static const int OVERFLOW_BIT = NUM_BITS_ENERGY_DATA - 1;

        static const int Emax = (1<<NUM_BITS_ENERGY_DATA);
        static const int signedEmax = (Emax>>1);

	// input data - need to confirm number of bits!
        typedef bitset<NUM_BITS_ENERGY_DATA> InputEnergyType;
	vector<InputEnergyType> inputEx;
	vector<InputEnergyType> inputEy;
	vector<InputEnergyType> inputEt;
	
	// output data
	bitset<NUM_BITS_ENERGY_DATA> outputEx;
	bitset<NUM_BITS_ENERGY_DATA> outputEy;
	bitset<NUM_BITS_ENERGY_DATA> outputEt;
	
	
};

#endif /*L1GCTWHEELENERGYFPGA_H_*/

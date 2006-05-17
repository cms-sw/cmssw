#ifndef L1GCTWHEELENERGYFPGA_H_
#define L1GCTWHEELENERGYFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <bitset>
#include <vector>

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
	///	
	/// get input data
	inline L1GctEtComponent inputEx(unsigned leafnum) const { return m_inputEx[leafnum]; }
	inline L1GctEtComponent inputEy(unsigned leafnum) const { return m_inputEy[leafnum]; }
	inline L1GctScalarEtVal inputEt(unsigned leafnum) const { return m_inputEt[leafnum]; }
	///
	/// get output data
	inline L1GctEtComponent outputEx() const { return m_outputEx; }
	inline L1GctEtComponent outputEy() const { return m_outputEy; }
	inline L1GctScalarEtVal outputEt() const { return m_outputEt; }

private:

	///
	/// algo ID
	int m_id;
	///
	/// the jet leaf card
	std::vector<L1GctJetLeafCard*> m_inputLeafCards;

/*         typedef L1GctTwosComplement<12> L1GctEtComponent; */

/* 	static const int NUM_BITS_ENERGY_DATA = 13; */
/* 	static const int OVERFLOW_BIT = NUM_BITS_ENERGY_DATA - 1; */

/*         static const int Emax = (1<<NUM_BITS_ENERGY_DATA); */
/*         static const int signedEmax = (Emax>>1); */

/* 	// input data - need to confirm number of bits! */
/*         typedef std::bitset<NUM_BITS_ENERGY_DATA> InputEnergyType; */
	std::vector<L1GctEtComponent> m_inputEx;
	std::vector<L1GctEtComponent> m_inputEy;
	std::vector<L1GctScalarEtVal> m_inputEt;
	
	// output data
	L1GctEtComponent m_outputEx;
	L1GctEtComponent m_outputEy;
	L1GctScalarEtVal m_outputEt;
	
	
};

#endif /*L1GCTWHEELENERGYFPGA_H_*/

#ifndef L1GCTWHEELENERGYFPGA_H_
#define L1GCTWHEELENERGYFPGA_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <vector>

class L1GctJetLeafCard;

class L1GctWheelEnergyFpga : public L1GctProcessor
{
public:
	L1GctWheelEnergyFpga(int id, std::vector<L1GctJetLeafCard*> leafCards);
	~L1GctWheelEnergyFpga();
	///

        /// Overload << operator
        friend std::ostream& operator << (std::ostream& os, const L1GctWheelEnergyFpga& fpga);

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
	void setInputEnergy(int i, int ex, int ey, unsigned et);
	///	
	/// get input data
	inline L1GctEtComponent getInputEx(unsigned leafnum) const { return m_inputEx[leafnum]; }
	inline L1GctEtComponent getInputEy(unsigned leafnum) const { return m_inputEy[leafnum]; }
	inline L1GctScalarEtVal getInputEt(unsigned leafnum) const { return m_inputEt[leafnum]; }
	///
	/// get output data
	inline L1GctEtComponent getOutputEx() const { return m_outputEx; }
	inline L1GctEtComponent getOutputEy() const { return m_outputEy; }
	inline L1GctScalarEtVal getOutputEt() const { return m_outputEt; }

private:

	///
	/// algo ID
	int m_id;
	///
	/// the jet leaf card
	std::vector<L1GctJetLeafCard*> m_inputLeafCards;
	///
	/// the input components from each input card
	std::vector<L1GctEtComponent> m_inputEx;
	std::vector<L1GctEtComponent> m_inputEy;
	std::vector<L1GctScalarEtVal> m_inputEt;
	///
	/// output data
	L1GctEtComponent m_outputEx;
	L1GctEtComponent m_outputEy;
	L1GctScalarEtVal m_outputEt;
	
	
};

std::ostream& operator << (std::ostream& os, const L1GctWheelEnergyFpga& fpga);

#endif /*L1GCTWHEELENERGYFPGA_H_*/

#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <vector>

class L1GctWheelEnergyFpga;
class L1GctWheelJetFpga;
class L1GctJetFinalStage;

/*
 * Emulates the GCT global energy algorithms
 * author: Jim Brooke & Greg Heath
 * date: 20/2/2006
 * 
 */

class L1GctGlobalEnergyAlgos : public L1GctProcessor
{
public:
	L1GctGlobalEnergyAlgos();
	~L1GctGlobalEnergyAlgos();
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
	void setPlusWheelEnergyFpga (L1GctWheelEnergyFpga* fpga);
	void setMinusWheelEnergyFpga(L1GctWheelEnergyFpga* fpga);
	void setPlusWheelJetFpga (L1GctWheelJetFpga* fpga);
	void setMinusWheelJetFpga(L1GctWheelJetFpga* fpga);
	void setJetFinalStage(L1GctJetFinalStage* fpga);
	///	
	/// set input data per wheel
	void setInputWheelEx(unsigned wheel, int energy, bool overflow);
	void setInputWheelEy(unsigned wheel, int energy, bool overflow);
	void setInputWheelEt(unsigned wheel, unsigned energy, bool overflow);
	void setInputWheelHt(unsigned wheel, unsigned energy, bool overflow);

        // An extra contribution to Ht from jets at
        // the boundary between wheels
        void setInputBoundaryHt(unsigned energy, bool overflow);

        // also jet counts
        void setInputWheelJc(unsigned wheel, unsigned jcnum, unsigned count);
        void setInputBoundaryJc(unsigned jcnum, unsigned count);

	// return input data
        inline L1GctEtComponent inputExValPlusWheel() const { return m_exValPlusWheel; }
        inline L1GctEtComponent inputEyValPlusWheel() const { return m_eyValPlusWheel; }
        inline L1GctEtComponent inputExVlMinusWheel() const { return m_exVlMinusWheel; }
        inline L1GctEtComponent inputEyVlMinusWheel() const { return m_eyVlMinusWheel; }
	inline L1GctScalarEtVal inputEtValPlusWheel() const { return m_etValPlusWheel; }
	inline L1GctScalarEtVal inputHtValPlusWheel() const { return m_htValPlusWheel; }
	inline L1GctScalarEtVal inputEtVlMinusWheel() const { return m_etVlMinusWheel; }
	inline L1GctScalarEtVal inputHtVlMinusWheel() const { return m_htVlMinusWheel; }
	inline L1GctScalarEtVal inputHtBoundaryJets() const { return m_htBoundaryJets; }
        inline L1GctJcWheelType inputJcValPlusWheel(unsigned jcnum) {return m_jcValPlusWheel[jcnum]; }
        inline L1GctJcWheelType inputJcVlMinusWheel(unsigned jcnum) {return m_jcVlMinusWheel[jcnum]; }
        inline L1GctJcBoundType inputJcBoundaryJets(unsigned jcnum) {return m_jcBoundaryJets[jcnum]; }

	// return output data
	inline L1GctScalarEtVal etMiss()    { return m_outputEtMiss; }
	inline L1GctEtAngleBin  etMissPhi() { return m_outputEtMissPhi; }
	inline L1GctScalarEtVal etSum()     { return m_outputEtSum; }
	inline L1GctScalarEtVal etHad()     { return m_outputEtHad; }
	inline L1GctJcFinalType jetCount(unsigned jcnum) { return m_outputJetCounts[jcnum]; }
	
private:
	
	// Here are the algorithm types we get our inputs from
	L1GctWheelEnergyFpga* m_plusWheelFpga;
	L1GctWheelEnergyFpga* m_minusWheelFpga;
	L1GctWheelJetFpga* m_plusWheelJetFpga;
	L1GctWheelJetFpga* m_minusWheelJetFpga;
	L1GctJetFinalStage* m_jetFinalStage;

	// input data
	L1GctEtComponent m_exValPlusWheel;
        L1GctEtComponent m_eyValPlusWheel;
	L1GctScalarEtVal m_etValPlusWheel;
	L1GctScalarEtVal m_htValPlusWheel;
	L1GctEtComponent m_exVlMinusWheel;
	L1GctEtComponent m_eyVlMinusWheel;
	L1GctScalarEtVal m_etVlMinusWheel;
	L1GctScalarEtVal m_htVlMinusWheel;
        L1GctScalarEtVal m_htBoundaryJets;

        std::vector<L1GctJcWheelType> m_jcValPlusWheel;
        std::vector<L1GctJcWheelType> m_jcVlMinusWheel;
        std::vector<L1GctJcBoundType> m_jcBoundaryJets;

	// output data
	L1GctScalarEtVal m_outputEtMiss;
	L1GctEtAngleBin  m_outputEtMissPhi;
	L1GctScalarEtVal m_outputEtSum;
	L1GctScalarEtVal m_outputEtHad;
        std::vector<L1GctJcFinalType> m_outputJetCounts;

        // PRIVATE MEMBER FUNCTION
	// the Etmiss algorithm
        struct etmiss_vec {
	  L1GctScalarEtVal mag;
	  L1GctEtAngleBin  phi;
	};
        etmiss_vec calculate_etmiss_vec (L1GctEtComponent ex, L1GctEtComponent ey) ;
	
};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

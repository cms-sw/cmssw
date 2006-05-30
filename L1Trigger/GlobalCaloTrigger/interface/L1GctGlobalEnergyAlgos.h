#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <vector>

class L1GctWheelEnergyFpga;
class L1GctWheelJetFpga;

/*
 * Emulates the GCT global energy algorithms
 * author: Jim Brooke & Greg Heath
 * date: 20/2/2006
 * 
 */

class L1GctGlobalEnergyAlgos : public L1GctProcessor
{
public:
	L1GctGlobalEnergyAlgos(std::vector<L1GctWheelEnergyFpga*> WheelFpga,
			       std::vector<L1GctWheelJetFpga*> WheelJetFpga);
	~L1GctGlobalEnergyAlgos();
	///

        /// Overload << operator
        friend std::ostream& operator << (std::ostream& os, const L1GctGlobalEnergyAlgos& fpga);

	/// clear internal buffers
	virtual void reset();
	///
	/// get input data from sources
	virtual void fetchInput();
	///
	/// process the data, fill output buffers
	virtual void process();
	///	
	/// set input data per wheel
	void setInputWheelEx(unsigned wheel, int energy, bool overflow);
	void setInputWheelEy(unsigned wheel, int energy, bool overflow);
	void setInputWheelEt(unsigned wheel, unsigned energy, bool overflow);
	void setInputWheelHt(unsigned wheel, unsigned energy, bool overflow);

        // also jet counts
        void setInputWheelJc(unsigned wheel, unsigned jcnum, unsigned count);

	// return input data
        inline L1GctEtComponent getInputExValPlusWheel() const { return m_exValPlusWheel; }
        inline L1GctEtComponent getInputEyValPlusWheel() const { return m_eyValPlusWheel; }
        inline L1GctEtComponent getInputExVlMinusWheel() const { return m_exVlMinusWheel; }
        inline L1GctEtComponent getInputEyVlMinusWheel() const { return m_eyVlMinusWheel; }
	inline L1GctScalarEtVal getInputEtValPlusWheel() const { return m_etValPlusWheel; }
	inline L1GctScalarEtVal getInputHtValPlusWheel() const { return m_htValPlusWheel; }
	inline L1GctScalarEtVal getInputEtVlMinusWheel() const { return m_etVlMinusWheel; }
	inline L1GctScalarEtVal getInputHtVlMinusWheel() const { return m_htVlMinusWheel; }
        inline L1GctJcWheelType getInputJcValPlusWheel(unsigned jcnum) {return m_jcValPlusWheel[jcnum]; }
        inline L1GctJcWheelType getInputJcVlMinusWheel(unsigned jcnum) {return m_jcVlMinusWheel[jcnum]; }

	// return output data
	inline L1GctScalarEtVal getEtMiss()    { return m_outputEtMiss; }
	inline L1GctEtAngleBin  getEtMissPhi() { return m_outputEtMissPhi; }
	inline L1GctScalarEtVal getEtSum()     { return m_outputEtSum; }
	inline L1GctScalarEtVal getEtHad()     { return m_outputEtHad; }
	inline L1GctJcFinalType getJetCount(unsigned jcnum) { return m_outputJetCounts[jcnum]; }
	
private:
	
	// Here are the algorithm types we get our inputs from
	L1GctWheelEnergyFpga* m_plusWheelFpga;
	L1GctWheelEnergyFpga* m_minusWheelFpga;
	L1GctWheelJetFpga* m_plusWheelJetFpga;
	L1GctWheelJetFpga* m_minusWheelJetFpga;

	// input data
	L1GctEtComponent m_exValPlusWheel;
        L1GctEtComponent m_eyValPlusWheel;
	L1GctScalarEtVal m_etValPlusWheel;
	L1GctScalarEtVal m_htValPlusWheel;
	L1GctEtComponent m_exVlMinusWheel;
	L1GctEtComponent m_eyVlMinusWheel;
	L1GctScalarEtVal m_etVlMinusWheel;
	L1GctScalarEtVal m_htVlMinusWheel;

        std::vector<L1GctJcWheelType> m_jcValPlusWheel;
        std::vector<L1GctJcWheelType> m_jcVlMinusWheel;

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

std::ostream& operator << (std::ostream& os, const L1GctGlobalEnergyAlgos& fpga);

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

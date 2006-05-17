#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <bitset>
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
	inline unsigned long getInputHtValPlusWheel() { return inputHtValPlusWheel.to_ulong();; }
	inline L1GctScalarEtVal inputEtVlMinusWheel() const { return m_etVlMinusWheel; }
	inline unsigned long getInputHtVlMinusWheel() { return inputHtVlMinusWheel.to_ulong();; }
	inline unsigned long getInputHtBoundaryJets() { return inputHtBoundaryJets.to_ulong();; }
        inline unsigned long getInputJcValPlusWheel(unsigned jcnum) {return inputJcValPlusWheel[jcnum].to_ulong();; }
        inline unsigned long getInputJcVlMinusWheel(unsigned jcnum) {return inputJcVlMinusWheel[jcnum].to_ulong();; }
        inline unsigned long getInputJcBoundaryJets(unsigned jcnum) {return inputJcBoundaryJets[jcnum].to_ulong();; }

	// return output data
	inline L1GctScalarEtVal etMiss()    { return m_outputEtMiss; }
	inline L1GctEtAngleBin  etMissPhi() { return m_outputEtMissPhi; }
	inline L1GctScalarEtVal etSum()     { return m_outputEtSum; }
	inline unsigned long getEtHad()     { return outputEtHad.to_ulong(); }
	inline unsigned long getJetCounts(unsigned jcnum) { return outputJetCounts[jcnum].to_ulong(); }
	
private:
	
	// Here are the algorithm types we get our inputs from
	L1GctWheelEnergyFpga* m_plusWheelFpga;
	L1GctWheelEnergyFpga* m_minusWheelFpga;
	L1GctWheelJetFpga* m_plusWheelJetFpga;
	L1GctWheelJetFpga* m_minusWheelJetFpga;
	L1GctJetFinalStage* m_jetFinalStage;

        typedef std::bitset<3> JcBoundType;
        typedef std::bitset<3> JcWheelType;
        typedef std::bitset<5> JcFinalType;
	// input data - need to confirm number of bits!
	L1GctEtComponent m_exValPlusWheel;
        L1GctEtComponent m_eyValPlusWheel;
	L1GctScalarEtVal m_etValPlusWheel;
	std::bitset<12> inputHtValPlusWheel;
	L1GctEtComponent m_exVlMinusWheel;
	L1GctEtComponent m_eyVlMinusWheel;
	L1GctScalarEtVal m_etVlMinusWheel;
	std::bitset<12> inputHtVlMinusWheel;
        std::bitset<12> inputHtBoundaryJets;

        bool ovfloHtValPlusWheel;
        bool ovfloHtVlMinusWheel;
        bool ovfloHtBoundaryJets;

        std::vector<JcWheelType> inputJcValPlusWheel;
        std::vector<JcWheelType> inputJcVlMinusWheel;
        std::vector<JcBoundType> inputJcBoundaryJets;

        // internal stuff for inputs and outputs
        void checkUnsignedNatural(  unsigned E, bool O, int nbits, unsigned long &Eout, bool &Oout);
	void decodeUnsignedInput( unsigned long Ein, unsigned &Eout, bool &Oout);
        // internal stuff for the Etmiss algorithm
        struct etmiss_vec {
	  L1GctScalarEtVal mag;
	  L1GctEtAngleBin  phi;
	};
        etmiss_vec calculate_etmiss_vec (L1GctEtComponent ex, L1GctEtComponent ey) ;
	
	// output data
	L1GctScalarEtVal m_outputEtMiss;
	L1GctEtAngleBin  m_outputEtMissPhi;
	L1GctScalarEtVal m_outputEtSum;
	std::bitset<13> outputEtHad;
        std::vector<JcFinalType> outputJetCounts;

};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

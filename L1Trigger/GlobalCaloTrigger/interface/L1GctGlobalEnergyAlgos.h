#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <bitset>
#include <vector>

using namespace std;

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
        long int getInputExValPlusWheel();
        long int getInputEyValPlusWheel();
        long int getInputExVlMinusWheel();
        long int getInputEyVlMinusWheel();
	inline unsigned long getInputEtValPlusWheel() { return inputEtValPlusWheel.to_ulong();; }
	inline unsigned long getInputHtValPlusWheel() { return inputHtValPlusWheel.to_ulong();; }
	inline unsigned long getInputEtVlMinusWheel() { return inputEtVlMinusWheel.to_ulong();; }
	inline unsigned long getInputHtVlMinusWheel() { return inputHtVlMinusWheel.to_ulong();; }
	inline unsigned long getInputHtBoundaryJets() { return inputHtBoundaryJets.to_ulong();; }
        inline unsigned long getInputJcValPlusWheel(unsigned jcnum) {return inputJcValPlusWheel[jcnum].to_ulong();; }
        inline unsigned long getInputJcVlMinusWheel(unsigned jcnum) {return inputJcVlMinusWheel[jcnum].to_ulong();; }
        inline unsigned long getInputJcBoundaryJets(unsigned jcnum) {return inputJcBoundaryJets[jcnum].to_ulong();; }

	// return output data
	inline unsigned long getEtMiss()    { return outputEtMiss.to_ulong(); }
	inline unsigned long getEtMissPhi() { return outputEtMissPhi.to_ulong(); }
	inline unsigned long getEtSum()     { return outputEtSum.to_ulong(); }
	inline unsigned long getEtHad()     { return outputEtHad.to_ulong(); }
	inline unsigned long getJetCounts(unsigned jcnum) { return outputJetCounts[jcnum].to_ulong(); }
	
private:
	
	// Here are the algorithm types we get our inputs from
	L1GctWheelEnergyFpga* m_plusWheelFpga;
	L1GctWheelEnergyFpga* m_minusWheelFpga;
	L1GctWheelJetFpga* m_plusWheelJetFpga;
	L1GctWheelJetFpga* m_minusWheelJetFpga;
	L1GctJetFinalStage* m_jetFinalStage;

        typedef bitset<3> JcBoundType;
        typedef bitset<3> JcWheelType;
        typedef bitset<5> JcFinalType;
	// input data - need to confirm number of bits!
	bitset<12> inputExValPlusWheel;
	bitset<12> inputEyValPlusWheel;
	bitset<12> inputEtValPlusWheel;
	bitset<12> inputHtValPlusWheel;
	bitset<12> inputExVlMinusWheel;
	bitset<12> inputEyVlMinusWheel;
	bitset<12> inputEtVlMinusWheel;
	bitset<12> inputHtVlMinusWheel;
        bitset<12> inputHtBoundaryJets;

        bool ovfloExValPlusWheel;
        bool ovfloEyValPlusWheel;
        bool ovfloEtValPlusWheel;
        bool ovfloHtValPlusWheel;
        bool ovfloExVlMinusWheel;
        bool ovfloEyVlMinusWheel;
        bool ovfloEtVlMinusWheel;
        bool ovfloHtVlMinusWheel;
        bool ovfloHtBoundaryJets;

        vector<JcWheelType> inputJcValPlusWheel;
        vector<JcWheelType> inputJcVlMinusWheel;
        vector<JcBoundType> inputJcBoundaryJets;

        // internal stuff for inputs and outputs
        void checkUnsignedNatural(  unsigned E, bool O, int nbits, unsigned long &Eout, bool &Oout);
	void checkIntegerTwosComplement( int E, bool O, int nbits, unsigned long &Eout, bool &Oout);
	void decodeUnsignedInput( unsigned long Ein, unsigned &Eout, bool &Oout);
	void decodeIntegerInput ( unsigned long Ein, int &Eout, bool &Oout);
        long int longIntegerFromTwosComplement (bitset<12> energyBits);
        // internal stuff for the Etmiss algorithm
        struct etmiss_vec { unsigned long mag; unsigned phi;};
        etmiss_vec calculate_etmiss_vec (long int Ex, long int Ey) ;
	
	// output data
	bitset<13> outputEtMiss;
	bitset<7> outputEtMissPhi;
	bitset<13> outputEtSum;
	bitset<13> outputEtHad;
        vector<JcFinalType> outputJetCounts;

};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

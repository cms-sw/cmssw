#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include <bitset>
#include <vector>

using namespace std;

/*
 * Emulates the GCT global energy algorithms
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctGlobalEnergyAlgos
{
public:
	L1GctGlobalEnergyAlgos();
	~L1GctGlobalEnergyAlgos();

	// clear internal data
	void reset();
	
	// process the event
	void process();

	// set input data per wheel
/* 	void setInputWheelEx(unsigned wheel, int energy, bool overflow); */
/* 	void setInputWheelEy(unsigned wheel, int energy, bool overflow); */
	void setInputWheelEt(unsigned wheel, unsigned energy, bool overflow);
	void setInputWheelHt(unsigned wheel, unsigned energy, bool overflow);

        // An extra contribution to Ht from jets at
        // the boundary between wheels
        void setInputBoundaryHt(unsigned energy, bool overflow);

        // also jet counts
        void setInputWheelJc(unsigned wheel, unsigned jcnum, unsigned count);
        void setInputBoundaryJc(unsigned jcnum, unsigned count);

/* 	// return input data */
/*         long int getInputExValPlusWheel(); */
/*         long int getInputEyValPlusWheel(); */
/*         long int getInputExVlMinusWheel(); */
/*         long int getInputEyVlMinusWheel(); */
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
	
        typedef bitset<3> JcBoundType;
        typedef bitset<4> JcWheelType;
        typedef bitset<5> JcFinalType;
	// input data - need to confirm number of bits!
/* 	bitset<12> inputExValPlusWheel; */
/* 	bitset<12> inputEyValPlusWheel; */
	bitset<12> inputEtValPlusWheel;
	bitset<12> inputHtValPlusWheel;
/* 	bitset<12> inputExVlMinusWheel; */
/* 	bitset<12> inputEyVlMinusWheel; */
	bitset<12> inputEtVlMinusWheel;
	bitset<12> inputHtVlMinusWheel;
        bitset<12> inputHtBoundaryJets;

/*         bool ovfloExValPlusWheel; */
/*         bool ovfloEyValPlusWheel; */
        bool ovfloEtValPlusWheel;
        bool ovfloHtValPlusWheel;
/*         bool ovfloExVlMinusWheel; */
/*         bool ovfloEyVlMinusWheel; */
        bool ovfloEtVlMinusWheel;
        bool ovfloHtVlMinusWheel;
        bool ovfloHtBoundaryJets;

        vector<JcWheelType> inputJcValPlusWheel;
        vector<JcWheelType> inputJcVlMinusWheel;
        vector<JcBoundType> inputJcBoundaryJets;
	
	// output data
	bitset<13> outputEtMiss;
	bitset<6> outputEtMissPhi;
	bitset<13> outputEtSum;
	bitset<13> outputEtHad;
        vector<JcFinalType> outputJetCounts;

};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

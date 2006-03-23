#ifndef L1GCTGLOBALENERGYALGOS_H_
#define L1GCTGLOBALENERGYALGOS_H_

#include <bitset>

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
/* 	void setInputWheelEt(unsigned wheel, unsigned energy, bool overflow); */
	void setInputWheelHt(unsigned wheel, unsigned energy, bool overflow);

        // An extra contribution to Ht from jets at
        // the boundary between wheels
        void setInputBoundaryHt(unsigned energy, bool overflow);

/*         // also jet counts */
/*         void setInputWheelJc(unsigned wheel, unsigned jcnum, unsigned count); */
/*         void setInputBoundaryJc(unsigned jcnum, unsigned count); */

/* 	// return input data */
/*         long int getInputExForwardWheel(); */
/*         long int getInputEyForwardWheel(); */
/*         long int getInputExBakwardWheel(); */
/*         long int getInputEyBakwardWheel(); */
/* 	inline unsigned long getInputEtForwardWheel() { return inputEtForwardWheel.to_ulong();; } */
	inline unsigned long getInputHtForwardWheel() { return inputHtForwardWheel.to_ulong();; }
/* 	inline unsigned long getInputEtBakwardWheel() { return inputEtBakwardWheel.to_ulong();; } */
	inline unsigned long getInputHtBakwardWheel() { return inputHtBakwardWheel.to_ulong();; }
	inline unsigned long getInputHtBoundaryJets() { return inputHtBoundaryJets.to_ulong();; }
/*         inline unsigned long getInputJcForwardWheel(unsigned jcnum) {return inputJcForwardWheel[jcnum].to_ulong();; } */
/*         inline unsigned long getInputJcBakwardWheel(unsigned jcnum) {return inputJcBakwardWheel[jcnum].to_ulong();; } */
/*         inline unsigned long getInputJcBoundaryJets(unsigned jcnum) {return inputJcBoundaryJets[jcnum].to_ulong();; } */

/* 	// return output data	 */
	inline unsigned long getEtMiss()    { return outputEtMiss.to_ulong(); }
	inline unsigned long getEtMissPhi() { return outputEtMissPhi.to_ulong(); }
	inline unsigned long getEtSum()     { return outputEtSum.to_ulong(); }
	inline unsigned long getEtHad()     { return outputEtHad.to_ulong(); }
/* 	inline unsigned long getJetCounts(unsigned jcnum) { return output[jcnum].to_ulong(); } */
	
private:
	
	// input data - need to confirm number of bits!
/* 	bitset<12> inputExForwardWheel; */
/* 	bitset<12> inputEyForwardWheel; */
/* 	bitset<12> inputEtForwardWheel; */
	bitset<12> inputHtForwardWheel;
/* 	bitset<12> inputExBakwardWheel; */
/* 	bitset<12> inputEyBakwardWheel; */
/* 	bitset<12> inputEtBakwardWheel; */
	bitset<12> inputHtBakwardWheel;
        bitset<12> inputHtBoundaryJets;

/*         bool ovfloExForwardWheel; */
/*         bool ovfloEyForwardWheel; */
/*         bool ovfloEtForwardWheel; */
        bool ovfloHtForwardWheel;
/*         bool ovfloExBakwardWheel; */
/*         bool ovfloEyBakwardWheel; */
/*         bool ovfloEtBakwardWheel; */
        bool ovfloHtBakwardWheel;
        bool ovfloHtBoundaryJets;

/*         vector<bitset<4>> inputJcForwardWheel; */
/*         vector<bitset<4>> inputJcBakwardWheel; */
/*         vector<bitset<3>> inputJcBoundaryJets; */
	
	// output data
	bitset<13> outputEtMiss;
	bitset<6> outputEtMissPhi;
	bitset<13> outputEtSum;
	bitset<13> outputEtHad;
/*         vector<bitset<5>> outputJetCounts; */

};

#endif /*L1GCTGLOBALENERGYALGOS_H_*/

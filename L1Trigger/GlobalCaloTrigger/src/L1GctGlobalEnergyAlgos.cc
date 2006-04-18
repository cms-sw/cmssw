#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"

L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() :
  inputJcValPlusWheel(12),
  inputJcVlMinusWheel(12),
  inputJcBoundaryJets(12),
  outputJetCounts(12)
{
}

L1GctGlobalEnergyAlgos::~L1GctGlobalEnergyAlgos()
{
}

// clear internal data
void L1GctGlobalEnergyAlgos::reset()
{
  inputEtValPlusWheel.reset();
  inputEtVlMinusWheel.reset();
  inputHtValPlusWheel.reset();
  inputHtVlMinusWheel.reset();
  inputHtBoundaryJets.reset();
  for (int i=0; i<12; i++) {
    inputJcValPlusWheel[i].reset();
    inputJcVlMinusWheel[i].reset();
    inputJcBoundaryJets[i].reset();
  }
  //
  ovfloEtValPlusWheel = false;
  ovfloEtVlMinusWheel = false;
  ovfloHtValPlusWheel = false;
  ovfloHtVlMinusWheel = false;
  ovfloHtBoundaryJets = false;
  //
  outputEtSum.reset();
  outputEtHad.reset();
  for (int i=0; i<12; i++) {
    outputJetCounts[i].reset();
  }
}

void L1GctGlobalEnergyAlgos::fetchInput() {
	
}

// process the event
void L1GctGlobalEnergyAlgos::process()
{
  unsigned long EtPlus, EtMinus, EtSum;
  unsigned long HtPlus, HtMinus, HtBound, HtSum;
  bool HtOvflo;
  bool EtOvflo;
  bitset<13> EtResult;
  bitset<13> HtResult;

  // Form the Et sum
  EtPlus  = inputEtValPlusWheel.to_ulong();
  EtMinus = inputEtVlMinusWheel.to_ulong();
  //
  EtSum = EtPlus + EtMinus;
  if (EtSum>0xfff) {
    EtSum = EtSum % 0x1000;
    EtOvflo = true;
  } else {
    EtOvflo = ovfloEtValPlusWheel or ovfloEtVlMinusWheel;
  }
  //
  bitset<13> etBits(EtSum);
  EtResult = etBits;
  if (EtOvflo) {EtResult.set(12);}
  //
  outputEtSum = EtResult;

  // Form the Ht sum
  HtPlus  = inputHtValPlusWheel.to_ulong();
  HtMinus = inputHtVlMinusWheel.to_ulong();
  HtBound = inputHtBoundaryJets.to_ulong();
  //
  HtSum = HtPlus + HtMinus + HtBound;
  if (HtSum>0xfff) {
    HtSum = HtSum % 0x1000;
    HtOvflo = true;
  } else {
    HtOvflo = ovfloHtValPlusWheel or ovfloHtVlMinusWheel or ovfloHtBoundaryJets;
  }
  //
  bitset<13> htBits(HtSum);
  HtResult = htBits;
  if (HtOvflo) {HtResult.set(12);}
  //
  outputEtHad = HtResult;

  // Add the jet counts.
  // Use bitset operations to implement the addition.
  for (int i=0; i<12; i++) {
    JcFinalType jcResult;
    bool carry;
    //
    if ((inputJcValPlusWheel[i].count()==inputJcValPlusWheel[i].size()) ||
	(inputJcVlMinusWheel[i].count()==inputJcVlMinusWheel[i].size()) ||
	(inputJcBoundaryJets[i].count()==inputJcBoundaryJets[i].size())) {
      jcResult.set();
    } else {
      // Add the inputs from the two wheels
      jcResult.reset();
      carry = false;
      for (int j=0; j<4; j++) {
        bool b1, b2;
        b1 = inputJcValPlusWheel[i].test(j);
        b2 = inputJcVlMinusWheel[i].test(j);
        if ((b1 ^ b2) ^ carry) jcResult.set(j);
        carry = (b1&b2) | (b1&carry) | (b2&carry);
      }
      if (carry) jcResult.set(4);
      // Add the result to the boundary jet input
      carry = false;
      for (int j=0; j<3; j++) {
        bool b1, b2;
        b1 = inputJcBoundaryJets[i].test(j);
        b2 = jcResult.test(j);
        if ((b1 ^ b2) ^ carry) jcResult.set(j); else jcResult.reset(j);
        carry = (b1&b2) | (b1&carry) | (b2&carry);
      }
      for (int j=3; j<5; j++) {
	bool b;
	b = jcResult.test(j);
        if (carry ^ b) jcResult.set(j); else jcResult.reset(j);
        carry = carry & b;
      }
      if (carry) jcResult.set();
    }
    //
    outputJetCounts[i] = jcResult;
  }
}

// set input data per wheel
// void L1GctGlobalEnergyAlgos::setInputWheelEx(unsigned wheel, int energy, bool overflow)
// void L1GctGlobalEnergyAlgos::setInputWheelEy(unsigned wheel, int energy, bool overflow)
void L1GctGlobalEnergyAlgos::setInputWheelEt(unsigned wheel, unsigned energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  energyInput = energy;
  if (energyInput>0xfff) {
    energyInput = energyInput % 0x1000;
    energyOvflo = true;
  } else {
    energyOvflo = overflow;
  }
  bitset<12> energyBits(energyInput);
  if (wheel==0) {
    inputEtValPlusWheel = energyBits;
    ovfloEtValPlusWheel = energyOvflo;
  } else if (wheel==1) {
    inputEtVlMinusWheel = energyBits;
    ovfloEtVlMinusWheel = energyOvflo;
  }
}

void L1GctGlobalEnergyAlgos::setInputWheelHt(unsigned wheel, unsigned energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  energyInput = energy;
  if (energyInput>0xfff) {
    energyInput = energyInput % 0x1000;
    energyOvflo = true;
  } else {
    energyOvflo = overflow;
  }
  bitset<12> energyBits(energyInput);
  if (wheel==0) {
    inputHtValPlusWheel = energyBits;
    ovfloHtValPlusWheel = energyOvflo;
  } else if (wheel==1) {
    inputHtVlMinusWheel = energyBits;
    ovfloHtVlMinusWheel = energyOvflo;
  }
}


// Set the jet count input values
void L1GctGlobalEnergyAlgos::setInputWheelJc(unsigned wheel, unsigned jcnum, unsigned count)
{
  unsigned long valueInput;
  JcWheelType countBits;

  countBits.set();
  if (jcnum>=0 && jcnum<12) {
    valueInput = count;
    if (valueInput<0x10) {
     JcWheelType bits(valueInput);
     countBits = bits;
    }
    if (wheel==0) {
      inputJcValPlusWheel[jcnum] = countBits;
    } else if (wheel==1) {
      inputJcVlMinusWheel[jcnum] = countBits;
    }
  }

}


// An extra contribution to Ht from jets at
// the boundary between wheels
void L1GctGlobalEnergyAlgos::setInputBoundaryHt(unsigned energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  energyInput = energy;
  if (energyInput>0xfff) {
    energyInput = energyInput % 0x1000;
    energyOvflo = true;
  } else {
    energyOvflo = overflow;
  }
  bitset<12> energyBits(energyInput);
  inputHtBoundaryJets = energyBits;
  ovfloHtBoundaryJets = energyOvflo;
}


void L1GctGlobalEnergyAlgos::setInputBoundaryJc(unsigned jcnum, unsigned count)
{
  unsigned long valueInput;
  JcBoundType countBits;

  countBits.set();
  if (jcnum>=0 && jcnum<12) {
    valueInput = count;
    if (valueInput<0x08) {
     JcBoundType bits(valueInput);
     countBits = bits;
    }
    inputJcBoundaryJets[jcnum] = countBits;
  }

}


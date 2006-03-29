#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"

L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() // :
//   inputJcValPlusWheel(12),
//   inputJcVlMinusWheel(12),
//   inputJcBoundaryJets(12),
//   outputJetCounts(12)
{
}

L1GctGlobalEnergyAlgos::~L1GctGlobalEnergyAlgos()
{
}

// clear internal data
void L1GctGlobalEnergyAlgos::reset()
{
  inputHtValPlusWheel.reset();
  inputHtVlMinusWheel.reset();
  inputHtBoundaryJets.reset();
  //
  ovfloHtValPlusWheel = false;
  ovfloHtVlMinusWheel = false;
  ovfloHtBoundaryJets = false;
  //
  outputEtHad.reset();
}
	
// process the event
void L1GctGlobalEnergyAlgos::process()
{
  unsigned long HtPlus, HtMinus, HtBound, HtSum;
  bool HtOvflo;
  bitset<13> HtResult;

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
  HtResult.reset();
  for (int i=0;i<12;i++) {
    if ((HtSum & (1<<i)) != 0) {HtResult.set(i);}
  }
  if (HtOvflo) {HtResult.set(12);}
  //
  outputEtHad = HtResult;
}

// set input data per wheel
// void L1GctGlobalEnergyAlgos::setInputWheelEx(unsigned wheel, int energy, bool overflow)
// void L1GctGlobalEnergyAlgos::setInputWheelEy(unsigned wheel, int energy, bool overflow)
// void L1GctGlobalEnergyAlgos::setInputWheelEt(unsigned wheel, unsigned energy, bool overflow)
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



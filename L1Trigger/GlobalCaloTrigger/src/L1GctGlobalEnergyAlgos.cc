#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"

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
  inputExValPlusWheel.reset();
  inputExVlMinusWheel.reset();
  inputEyValPlusWheel.reset();
  inputEyVlMinusWheel.reset();
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
  ovfloExValPlusWheel = false;
  ovfloEyVlMinusWheel = false;
  ovfloExValPlusWheel = false;
  ovfloEyVlMinusWheel = false;
  ovfloEtValPlusWheel = false;
  ovfloEtVlMinusWheel = false;
  ovfloHtValPlusWheel = false;
  ovfloHtVlMinusWheel = false;
  ovfloHtBoundaryJets = false;
  //
  outputEtMiss.reset();
  outputEtMissPhi.reset();
  outputEtSum.reset();
  outputEtHad.reset();
  for (int i=0; i<12; i++) {
    outputJetCounts[i].reset();
  }
}

void L1GctGlobalEnergyAlgos::fetchInput() {
  unsigned EinU;
  int EinI;
  bool EOvflo;
  // input from WheelEnergyFpgas
  decodeIntegerInput ( m_plusWheelFpga->getOutputEx(), EinI, EOvflo);
  setInputWheelEx((unsigned) 0, EinI, EOvflo);
  decodeIntegerInput ( m_plusWheelFpga->getOutputEy(), EinI, EOvflo);
  setInputWheelEy((unsigned) 0, EinI, EOvflo);
  decodeUnsignedInput( m_plusWheelFpga->getOutputEt(), EinU, EOvflo);
  setInputWheelEt((unsigned) 0, EinU, EOvflo);
  //
  decodeIntegerInput ( m_minusWheelFpga->getOutputEx(), EinI, EOvflo);
  setInputWheelEx((unsigned) 1, EinI, EOvflo);
  decodeIntegerInput ( m_minusWheelFpga->getOutputEy(), EinI, EOvflo);
  setInputWheelEy((unsigned) 1, EinI, EOvflo);
  decodeUnsignedInput( m_minusWheelFpga->getOutputEt(), EinU, EOvflo);
  setInputWheelEt((unsigned) 1, EinU, EOvflo);
  //
  decodeUnsignedInput( m_plusWheelJetFpga->getOutputHt(), EinU, EOvflo);
  setInputWheelHt((unsigned) 0, EinU, EOvflo);
  decodeUnsignedInput( m_minusWheelJetFpga->getOutputHt(), EinU, EOvflo);
  setInputWheelHt((unsigned) 1, EinU, EOvflo);
  //
  decodeUnsignedInput( m_jetFinalStage->getHtBoundaryJets(), EinU, EOvflo);
  setInputBoundaryHt(EinU, EOvflo);
  for (unsigned i=0; i<12; i++) {
    setInputWheelJc((unsigned) 0, i, (unsigned) m_plusWheelJetFpga->getOutputJc(i));
    setInputWheelJc((unsigned) 1, i, (unsigned) m_minusWheelJetFpga->getOutputJc(i));
    setInputBoundaryJc(i, (unsigned) m_jetFinalStage->getJcBoundaryJets(i));
  }
}


// process the event
void L1GctGlobalEnergyAlgos::process()
{
  long int ExPlus, ExMinus, ExSum;
  long int EyPlus, EyMinus, EySum;
  bool ExOvflo, EyOvflo;
  L1GctGlobalEnergyAlgos::etmiss_vec EtMissing;
  unsigned long EtPlus, EtMinus, EtSum;
  unsigned long HtPlus, HtMinus, HtBound, HtSum;
  bool HtOvflo;
  bool EtOvflo;
  bitset<13> magResult;
  bitset<7>  phiResult;
  bitset<13> EtResult;
  bitset<13> HtResult;

  const unsigned Emax=(1<<12);
  const int signedEmax=Emax/2;

  //
  //-----------------------------------------------------------------------------
  // Form the Ex and Ey sums
  ExPlus  = longIntegerFromTwosComplement(inputExValPlusWheel);
  ExMinus = longIntegerFromTwosComplement(inputExVlMinusWheel);
  EyPlus  = longIntegerFromTwosComplement(inputEyValPlusWheel);
  EyMinus = longIntegerFromTwosComplement(inputEyVlMinusWheel);
  ExSum = ExPlus + ExMinus;
  if (ExSum>=signedEmax) {
    ExSum = ExSum - 2*signedEmax;
    ExOvflo = true;
  } else if (ExSum<-signedEmax) {
    ExSum = ExSum + 2*signedEmax;
    ExOvflo = true;
  } else {
    ExOvflo = ovfloExValPlusWheel or ovfloExVlMinusWheel;
  }
  EySum = EyPlus + EyMinus;
  if (EySum>=signedEmax) {
    EySum = EySum - 2*signedEmax;
    EyOvflo = true;
  } else if (EySum<-signedEmax) {
    EySum = EySum + 2*signedEmax;
    EyOvflo = true;
  } else {
    EyOvflo = ovfloEyValPlusWheel or ovfloEyVlMinusWheel;
  }
  // Execute the missing Et algorithm
  EtMissing = calculate_etmiss_vec(ExSum, EySum);
  //
  bitset<13> magBits(EtMissing.mag);
  magResult = magBits;
  if (ExOvflo or EyOvflo) {magResult.set(12);}
  bitset<7> phiBits(EtMissing.phi);
  phiResult = phiBits;
  //
  outputEtMiss = magResult;
  outputEtMissPhi = phiResult;

  //
  //-----------------------------------------------------------------------------
  // Form the Et sum
  EtPlus  = inputEtValPlusWheel.to_ulong();
  EtMinus = inputEtVlMinusWheel.to_ulong();
  //
  EtSum = EtPlus + EtMinus;
  if (EtSum>=Emax) {
    EtSum = EtSum % Emax;
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

  //
  //-----------------------------------------------------------------------------
  // Form the Ht sum
  HtPlus  = inputHtValPlusWheel.to_ulong();
  HtMinus = inputHtVlMinusWheel.to_ulong();
  HtBound = inputHtBoundaryJets.to_ulong();
  //
  HtSum = HtPlus + HtMinus + HtBound;
  if (HtSum>=Emax) {
    HtSum = HtSum % Emax;
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

  //
  //-----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------------------------
// set input data sources
//
void L1GctGlobalEnergyAlgos::setPlusWheelEnergyFpga (L1GctWheelEnergyFpga* fpga)
{
  m_plusWheelFpga = fpga;
}

void L1GctGlobalEnergyAlgos::setMinusWheelEnergyFpga(L1GctWheelEnergyFpga* fpga)
{
  m_minusWheelFpga = fpga;
}

void L1GctGlobalEnergyAlgos::setPlusWheelJetFpga (L1GctWheelJetFpga* fpga)
{
  m_plusWheelJetFpga = fpga;
}

void L1GctGlobalEnergyAlgos::setMinusWheelJetFpga(L1GctWheelJetFpga* fpga)
{
  m_minusWheelJetFpga = fpga;
}

void L1GctGlobalEnergyAlgos::setJetFinalStage(L1GctJetFinalStage* jfs)
{
    m_jetFinalStage = jfs;
}

//----------------------------------------------------------------------------------------------
// set input data per wheel: x component of missing Et
//
void L1GctGlobalEnergyAlgos::setInputWheelEx(unsigned wheel, int energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  checkIntegerTwosComplement(energy, overflow, (int) 12, energyInput, energyOvflo);

  bitset<12> energyBits(energyInput);
  if (wheel==0) {
    inputExValPlusWheel = energyBits;
    ovfloExValPlusWheel = energyOvflo;
  } else if (wheel==1) {
    inputExVlMinusWheel = energyBits;
    ovfloExVlMinusWheel = energyOvflo;
  }
}

//----------------------------------------------------------------------------------------------
// set input data per wheel: y component of missing Et
//
void L1GctGlobalEnergyAlgos::setInputWheelEy(unsigned wheel, int energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  checkIntegerTwosComplement(energy, overflow, (int) 12, energyInput, energyOvflo);

  bitset<12> energyBits(energyInput);
  if (wheel==0) {
    inputEyValPlusWheel = energyBits;
    ovfloEyValPlusWheel = energyOvflo;
  } else if (wheel==1) {
    inputEyVlMinusWheel = energyBits;
    ovfloEyVlMinusWheel = energyOvflo;
  }
}

//----------------------------------------------------------------------------------------------
// set input data per wheel: scalar sum of Et
//
void L1GctGlobalEnergyAlgos::setInputWheelEt(unsigned wheel, unsigned energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  checkUnsignedNatural(energy, overflow, (int) 12, energyInput, energyOvflo);

  bitset<12> energyBits(energyInput);
  if (wheel==0) {
    inputEtValPlusWheel = energyBits;
    ovfloEtValPlusWheel = energyOvflo;
  } else if (wheel==1) {
    inputEtVlMinusWheel = energyBits;
    ovfloEtVlMinusWheel = energyOvflo;
  }
}

//----------------------------------------------------------------------------------------------
// set input data per wheel: sum of transverse energy in jets (Ht)
//
void L1GctGlobalEnergyAlgos::setInputWheelHt(unsigned wheel, unsigned energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  checkUnsignedNatural(energy, overflow, (int) 12, energyInput, energyOvflo);

  bitset<12> energyBits(energyInput);
  if (wheel==0) {
    inputHtValPlusWheel = energyBits;
    ovfloHtValPlusWheel = energyOvflo;
  } else if (wheel==1) {
    inputHtVlMinusWheel = energyBits;
    ovfloHtVlMinusWheel = energyOvflo;
  }
}


//----------------------------------------------------------------------------------------------
// An extra contribution to Ht from jets at
// the boundary between wheels
//
void L1GctGlobalEnergyAlgos::setInputBoundaryHt(unsigned energy, bool overflow)
{
  unsigned long energyInput;
  bool          energyOvflo;

  checkUnsignedNatural(energy, overflow, (int) 12, energyInput, energyOvflo);

  bitset<12> energyBits(energyInput);
  inputHtBoundaryJets = energyBits;
  ovfloHtBoundaryJets = energyOvflo;
}


//----------------------------------------------------------------------------------------------
// Set the jet count input values
//
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


//----------------------------------------------------------------------------------------------
// Extra contributions to jet counts from jets at
// the boundary between wheels
//
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

//----------------------------------------------------------------------------------------------
// Return the stored input values (for signed quantities,
// i.e. Ex and Ey components) 
//
long int L1GctGlobalEnergyAlgos::getInputExValPlusWheel()
{
  return longIntegerFromTwosComplement(inputExValPlusWheel);
}
long int L1GctGlobalEnergyAlgos::getInputEyValPlusWheel()
{
  return longIntegerFromTwosComplement(inputEyValPlusWheel);
}
long int L1GctGlobalEnergyAlgos::getInputExVlMinusWheel()
{
  return longIntegerFromTwosComplement(inputExVlMinusWheel);
}
long int L1GctGlobalEnergyAlgos::getInputEyVlMinusWheel()
{
  return longIntegerFromTwosComplement(inputEyVlMinusWheel);
}

//----------------------------------------------------------------------------------------------
// The following code is private.
//
// Functions to convert input energies and overflow bits
// to unsigned quantities that can be stored in a given 
// number of bits.
//
// Separate versions for unsigned and integer inputs.
//
void L1GctGlobalEnergyAlgos::checkUnsignedNatural(  unsigned E, bool O, int nbits, unsigned long &Eout, bool &Oout)
{
  unsigned long energyInput;
  bool          energyOvflo;
  const unsigned max=(1<<nbits);

  energyInput = E;
  if (energyInput>=max) {
    energyInput = energyInput % max;
    energyOvflo = true;
  } else {
    energyOvflo = O;
  }
  Eout = energyInput;
  Oout = energyOvflo;
}

void L1GctGlobalEnergyAlgos::checkIntegerTwosComplement( int E, bool O, int nbits, unsigned long &Eout, bool &Oout)
{
  unsigned long energyInput;
  bool          energyOvflo;
  const unsigned max=(1<<(nbits-1));

  if (E>=0) {
    energyInput = E;
    if (energyInput>=max) {
      energyInput = energyInput % max;
      energyOvflo = true;
    } else {
      energyOvflo = O; // this is input argument 'O', not zero!
    }
  } else {
    int modE;
    const unsigned shift=(1<<nbits);
    modE = E + shift;
    energyOvflo = O; // this is input argument 'O', not zero!
    while (modE<0) {
      modE = modE + shift;
      energyOvflo = true;
    }
    energyInput=modE;
    if (energyInput<max) {
      energyInput = energyInput+max;
      energyOvflo = true;
    }
  }
  Eout = energyInput;
  Oout = energyOvflo;
}

//----------------------------------------------------------------------------------------------
//
// Decode 13-bit value to 12-bits unsigned plus overflow
void L1GctGlobalEnergyAlgos::decodeUnsignedInput( unsigned long Ein, unsigned &Eout, bool &Oout)
{
  unsigned energyInput;
  bool     energyOvflo;
  const unsigned max=(1<<12);

  energyInput = Ein;
  if (energyInput>=max) {
    energyInput = energyInput % max;
    energyOvflo = true;
  } else {
    energyOvflo = false;
  }
  Eout = energyInput;
  Oout = energyOvflo;

}

//
// Decode 13-bit value to 12-bits 2's complement plus overflow
void L1GctGlobalEnergyAlgos::decodeIntegerInput ( unsigned long Ein, int &Eout, bool &Oout)
{
  unsigned energyInput;
  bool     energyOvflo;
  int      energyValue;
  const unsigned max=(1<<12);

  energyInput = Ein;
  if (energyInput>=max) {
    energyInput = energyInput % max;
    energyOvflo = true;
  } else {
    energyOvflo = false;
  }
  if (energyInput>=max/2) {
    energyValue = (int) energyInput - (int) max;
  } else {
    energyValue = (int) energyInput;
  }

  Eout = energyValue;
  Oout = energyOvflo;

}

//----------------------------------------------------------------------------------------------
// Converts a value stored in a bitset in two's complement format
// to a (signed) integer.
//
long int L1GctGlobalEnergyAlgos::longIntegerFromTwosComplement (bitset<12> energyBits)
{
  long int e;
  const int max=(1<<(energyBits.size()-1));
  e = energyBits.to_ulong();
  if (e>=max) {
    e = e - 2*max;
  }
  return e;
}

//-----------------------------------------------------------------------------------
//
// Here's the Etmiss calculation
//
//-----------------------------------------------------------------------------------
L1GctGlobalEnergyAlgos::etmiss_vec
L1GctGlobalEnergyAlgos::calculate_etmiss_vec (long int Ex, long int Ey)
{
  //---------------------------------------------------------------------------------
  //
  // Calculates magnitude and direction of missing Et, given measured Ex and Ey.
  //
  // The algorithm used is suitable for implementation in hardware, using integer
  // multiplication, addition and comparison and bit shifting operations.
  //
  // Proceed in two stages. The first stage gives a result that lies between
  // 92% and 100% of the true Et, with the direction measured in 45 degree bins.
  // The final precision depends on the number of factors used in corrFact.
  // The present version with eleven factors gives a precision of 1% on Et, and
  // finds the direction to the nearest 5 degrees.
  //
  //---------------------------------------------------------------------------------
  etmiss_vec result;

  unsigned eneCoarse, phiCoarse;
  unsigned eneCorect, phiCorect;

  const unsigned root2fact = 181;
  const unsigned corrFact[11] = {24, 39, 51, 60, 69, 77, 83, 89, 95, 101, 106};
  const unsigned corrDphi[11] = { 0,  1,  2,  2,  3,  3,  3,  3,  4,   4,   4};

  vector<bool> s(3);
  unsigned Mx, My, Mw;

  unsigned Dx, Dy;
  unsigned eFact;

  unsigned b,phibin;
  bool midphi;

  // Here's the coarse calculation, with just one multiply operation
  //
  My = (unsigned) abs(Ey);
  Mx = (unsigned) abs(Ex);
  Mw = ((Mx+My)*root2fact)>>8;

  s[0] = (Ey<0);
  s[1] = (Ex<0);
  s[2] = (My>Mx);

  phibin = 0; b = 0;
  for (int i=0; i<3; i++) {
    if (s[i]) { b=1-b;} phibin = 2*phibin + b;
  }

  eneCoarse = max(max(Mx, My), Mw);
  phiCoarse = phibin*9;

  // For the fine calculation we multiply both input components
  // by all the factors in the corrFact list in order to find
  // the required corrections to the energy and angle
  //
  for (eFact=0; eFact<10; eFact++) {
    Dx = (Mx*corrFact[eFact])>>8;
    Dy = (My*corrFact[eFact])>>8;
    if         ((Dx>My) || (Dy>Mx))         {midphi=false; break;}
    if ((Mx+Dx)>(My-Dy) && (My+Dy)>(Mx-Dx)) {midphi=true;  break;}
  }
  eneCorect = (eneCoarse*(128+eFact))>>7;
  if (midphi ^ (b==1)) {
    phiCorect = phiCoarse + 8 - corrDphi[eFact];
  } else {
    phiCorect = phiCoarse + corrDphi[eFact];
  }

  // Store the result of the calculation
  //
  result.mag = eneCorect;
  result.phi = phiCorect;

  return result;
}


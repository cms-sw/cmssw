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
  m_exValPlusWheel.reset();
  m_exVlMinusWheel.reset();
  m_eyValPlusWheel.reset();
  m_eyVlMinusWheel.reset();
  m_etValPlusWheel.reset();
  m_etVlMinusWheel.reset();
  inputHtValPlusWheel.reset();
  inputHtVlMinusWheel.reset();
  inputHtBoundaryJets.reset();
  for (int i=0; i<12; i++) {
    inputJcValPlusWheel[i].reset();
    inputJcVlMinusWheel[i].reset();
    inputJcBoundaryJets[i].reset();
  }
  //
  ovfloHtValPlusWheel = false;
  ovfloHtVlMinusWheel = false;
  ovfloHtBoundaryJets = false;
  //
  m_outputEtMiss.reset();
  m_outputEtMissPhi.reset();
  m_outputEtSum.reset();
  outputEtHad.reset();
  for (int i=0; i<12; i++) {
    outputJetCounts[i].reset();
  }
}

void L1GctGlobalEnergyAlgos::fetchInput() {
  unsigned EinU;
  bool EOvflo;
  // input from WheelEnergyFpgas
  m_exValPlusWheel = m_plusWheelFpga->outputEx();
  m_eyValPlusWheel = m_plusWheelFpga->outputEy();
  m_etValPlusWheel = m_plusWheelFpga->outputEt();
  
  m_exVlMinusWheel = m_minusWheelFpga->outputEx();
  m_eyVlMinusWheel = m_minusWheelFpga->outputEy();
  m_etVlMinusWheel = m_minusWheelFpga->outputEt();
  // input from WheelJetFpgas and JetFinalStage
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
  L1GctEtComponent ExSum, EySum;
  L1GctGlobalEnergyAlgos::etmiss_vec EtMissing;
  unsigned long HtPlus, HtMinus, HtBound, HtSum;
  bool HtOvflo;
  std::bitset<13> HtResult;

  const unsigned Emax=(1<<12);
 
  //
  //-----------------------------------------------------------------------------
  // Form the Ex and Ey sums
  ExSum = m_exValPlusWheel + m_exVlMinusWheel;
  EySum = m_eyValPlusWheel + m_eyVlMinusWheel;
  // Execute the missing Et algorithm
  EtMissing = calculate_etmiss_vec(ExSum, EySum);
  //
  m_outputEtMiss    = EtMissing.mag;
  m_outputEtMissPhi = EtMissing.phi;

  //
  //-----------------------------------------------------------------------------
  // Form the Et sum
  m_outputEtSum = m_etValPlusWheel + m_etVlMinusWheel;

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
  std::bitset<13> htBits(HtSum);
  HtResult = htBits;
  if (HtOvflo) {HtResult.set(12);}
  //
  outputEtHad = HtResult;

  //
  //-----------------------------------------------------------------------------
  // Add the jet counts.
  // Use std::bitset operations to implement the addition.
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
  if (wheel==0) {
    m_exValPlusWheel.setValue(energy);
    m_exValPlusWheel.setOverFlow(overflow);
  } else if (wheel==1) {
    m_exVlMinusWheel.setValue(energy);
    m_exVlMinusWheel.setOverFlow(overflow);
  }
}

//----------------------------------------------------------------------------------------------
// set input data per wheel: y component of missing Et
//
void L1GctGlobalEnergyAlgos::setInputWheelEy(unsigned wheel, int energy, bool overflow)
{
  if (wheel==0) {
    m_eyValPlusWheel.setValue(energy);
    m_eyValPlusWheel.setOverFlow(overflow);
  } else if (wheel==1) {
    m_eyVlMinusWheel.setValue(energy);
    m_eyVlMinusWheel.setOverFlow(overflow);
  }
}

//----------------------------------------------------------------------------------------------
// set input data per wheel: scalar sum of Et
//
void L1GctGlobalEnergyAlgos::setInputWheelEt(unsigned wheel, unsigned energy, bool overflow)
{
  if (wheel==0) {
    m_etValPlusWheel.setValue(energy);
    m_etValPlusWheel.setOverFlow(overflow);
  } else if (wheel==1) {
    m_etVlMinusWheel.setValue(energy);
    m_etVlMinusWheel.setOverFlow(overflow);
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

  std::bitset<12> energyBits(energyInput);
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

  std::bitset<12> energyBits(energyInput);
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

//-----------------------------------------------------------------------------------
//
// Here's the Etmiss calculation
//
//-----------------------------------------------------------------------------------
L1GctGlobalEnergyAlgos::etmiss_vec
L1GctGlobalEnergyAlgos::calculate_etmiss_vec (L1GctEtComponent ex, L1GctEtComponent ey)
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

  std::vector<bool> s(3);
  unsigned Mx, My, Mw;

  unsigned Dx, Dy;
  unsigned eFact;

  unsigned b,phibin;
  bool midphi;

  // Here's the coarse calculation, with just one multiply operation
  //
  My = static_cast<unsigned>(abs(ey.value()));
  Mx = static_cast<unsigned>(abs(ex.value()));
  Mw = ((Mx+My)*root2fact)>>8;

  s[0] = (ey.value()<0);
  s[1] = (ex.value()<0);
  s[2] = (My>Mx);

  phibin = 0; b = 0;
  for (int i=0; i<3; i++) {
    if (s[i]) { b=1-b;} phibin = 2*phibin + b;
  }

  eneCoarse = std::max(std::max(Mx, My), Mw);
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
  result.mag.setValue(eneCorect);
  result.phi.setValue(phiCorect);

  result.mag.setOverFlow( result.mag.overFlow() || ex.overFlow() || ey.overFlow() );

  return result;
}


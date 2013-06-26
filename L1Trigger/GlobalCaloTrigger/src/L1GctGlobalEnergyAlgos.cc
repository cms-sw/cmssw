#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalHfSumAlgos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using std::ostream;
using std::endl;
using std::vector;
using std::max;

L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos(const std::vector<L1GctWheelEnergyFpga*>& wheelFpga,
					       const std::vector<L1GctWheelJetFpga*>& wheelJetFpga) :
  L1GctProcessor(),
  m_plusWheelFpga(wheelFpga.at(1)),
  m_minusWheelFpga(wheelFpga.at(0)),
  m_plusWheelJetFpga(wheelJetFpga.at(1)),
  m_minusWheelJetFpga(wheelJetFpga.at(0)),
  m_metComponents(0,0, L1GctMet::cordicTranslate),
  m_mhtComponents(0,0, L1GctMet::useHtMissLut),
  m_exValPlusWheel(), m_eyValPlusWheel(),
  m_etValPlusWheel(), m_htValPlusWheel(),
  m_exVlMinusWheel(), m_eyVlMinusWheel(),
  m_etVlMinusWheel(), m_htVlMinusWheel(),
  m_exValPlusPipe(), m_eyValPlusPipe(),
  m_etValPlusPipe(), m_htValPlusPipe(),
  m_exVlMinusPipe(), m_eyVlMinusPipe(),
  m_etVlMinusPipe(), m_htVlMinusPipe(),
  m_outputEtMiss(), m_outputEtMissPhi(),
  m_outputEtSum(), m_outputEtHad(),
  m_setupOk(true)
{
  if(wheelFpga.size() != 2)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() : Global Energy Algos has been incorrectly constructed!\n"
	  << "This class needs two wheel card pointers. "
	  << "Number of wheel card pointers present is " << wheelFpga.size() << ".\n";
      }
    }
  
  if(wheelJetFpga.size() != 2)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() : Global Energy Algos has been incorrectly constructed!\n"
	  << "This class needs two wheel jet fpga pointers. "
	  << "Number of wheel jet fpga pointers present is " << wheelJetFpga.size() << ".\n";
      }
    }
  
  if(m_plusWheelFpga == 0)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() has been incorrectly constructed!\n"
	  << "Plus Wheel Fpga pointer has not been set!\n";
      }
    }
  if(m_minusWheelFpga == 0)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() has been incorrectly constructed!\n"
	  << "Minus Wheel Fpga pointer has not been set!\n";
      }
    }
  if(m_plusWheelJetFpga == 0)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() has been incorrectly constructed!\n"
	  << "Plus Wheel Jet Fpga pointer has not been set!\n";
      }
    }
  if(m_minusWheelJetFpga == 0)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalEnergyAlgos::L1GctGlobalEnergyAlgos() has been incorrectly constructed!\n"
	  << "Minus Wheel Jet Fpga pointer has not been set!\n";
      }
    }

  // Set the scale for missing Et and missing Ht
  // Missing Et has one extra bit of precision added in the (Ex, Ey) 
  // conversion step, so we reverse this here.
  m_metComponents.setBitShift(1);
  // Missing Ht has its own bit shifting before the LUT, so we don't
  // need any extra
  m_mhtComponents.setBitShift(0);

  // Setup to perform the Hf sums
  m_hfSumProcessor = new L1GctGlobalHfSumAlgos(wheelJetFpga);

  m_setupOk &= m_hfSumProcessor->setupOk();

  if (!m_setupOk && m_verbose) {
    edm::LogError("L1GctSetupError") << "L1GctGlobalEnergyAlgos has been incorrectly constructed";
  }
}

L1GctGlobalEnergyAlgos::~L1GctGlobalEnergyAlgos()
{
  if (m_hfSumProcessor != 0) {
    delete m_hfSumProcessor;
  }
}

ostream& operator << (ostream& os, const L1GctGlobalEnergyAlgos& fpga)
{
  os << "===L1GctGlobalEnergyAlgos===" << endl;
  os << "WheelEnergyFpga* minus = " << fpga.m_minusWheelFpga << endl;
  os << "WheelEnergyFpga* plus  = " << fpga.m_plusWheelFpga << endl;
  os << "WheelJetFpga* minus = " << fpga.m_minusWheelJetFpga << endl;
  os << "WheelJetFpga* plus  = " << fpga.m_plusWheelJetFpga << endl;
  os << "Inputs from Plus wheel:" << endl;
  os << "  Ex " << fpga.m_exValPlusWheel << "\n  Ey " << fpga.m_eyValPlusWheel << endl;
  os << "  Et " << fpga.m_etValPlusWheel << "\n  Ht " << fpga.m_htValPlusWheel << endl; 
  os << "Inputs from Minus wheel:" << endl;
  os << "  Ex " << fpga.m_exVlMinusWheel << "\n  Ey " << fpga.m_eyVlMinusWheel << endl;
  os << "  Et " << fpga.m_etVlMinusWheel << "\n  Ht " << fpga.m_htVlMinusWheel << endl; 
  int bxZero = -fpga.bxMin();
  if (bxZero>=0 && bxZero<fpga.numOfBx()) {
    os << "Output Etmiss " << fpga.m_outputEtMiss.contents.at(bxZero) << endl;
    os << "Output Etmiss Phi " << fpga.m_outputEtMissPhi.contents.at(bxZero) << endl;
    os << "Output EtSum " << fpga.m_outputEtSum.contents.at(bxZero) << endl;
    os << "Output EtHad " << fpga.m_outputEtHad.contents.at(bxZero) << endl;
  }
  os << *fpga.m_hfSumProcessor;

  return os;
}

/// clear buffers
void L1GctGlobalEnergyAlgos::reset() {
  L1GctProcessor::reset();
  m_hfSumProcessor->reset();
}

/// partially clear buffers
void L1GctGlobalEnergyAlgos::setBxRange(const int firstBx, const int numberOfBx) {
  L1GctProcessor::setBxRange(firstBx, numberOfBx);
  m_hfSumProcessor->setBxRange(firstBx, numberOfBx);
}

void L1GctGlobalEnergyAlgos::setNextBx(const int bx) {
  L1GctProcessor::setNextBx(bx);
  m_hfSumProcessor->setNextBx(bx);
}

void L1GctGlobalEnergyAlgos::resetProcessor() {
  m_exValPlusWheel.reset();
  m_exVlMinusWheel.reset();
  m_eyValPlusWheel.reset();
  m_eyVlMinusWheel.reset();
  m_etValPlusWheel.reset();
  m_etVlMinusWheel.reset();
  m_htValPlusWheel.reset();
  m_htVlMinusWheel.reset();
  m_hxValPlusWheel.reset();
  m_hxVlMinusWheel.reset();
  m_hyValPlusWheel.reset();
  m_hyVlMinusWheel.reset();
}

void L1GctGlobalEnergyAlgos::resetPipelines() {
  m_outputEtMiss.reset    (numOfBx());
  m_outputEtMissPhi.reset (numOfBx());
  m_outputEtSum.reset     (numOfBx());
  m_outputEtHad.reset     (numOfBx());
  m_outputHtMiss.reset    (numOfBx());
  m_outputHtMissPhi.reset (numOfBx());

  m_exValPlusPipe.reset (numOfBx());
  m_eyValPlusPipe.reset (numOfBx());
  m_etValPlusPipe.reset (numOfBx());
  m_htValPlusPipe.reset (numOfBx());
  m_hxValPlusPipe.reset (numOfBx());
  m_hyValPlusPipe.reset (numOfBx());

  m_exVlMinusPipe.reset (numOfBx());
  m_eyVlMinusPipe.reset (numOfBx());
  m_etVlMinusPipe.reset (numOfBx());
  m_htVlMinusPipe.reset (numOfBx());
  m_hxVlMinusPipe.reset (numOfBx());
  m_hyVlMinusPipe.reset (numOfBx());

}

void L1GctGlobalEnergyAlgos::fetchInput() {
  if (m_setupOk) {
    // input from WheelEnergyFpgas
    m_exValPlusWheel = m_plusWheelFpga->getOutputEx();
    m_eyValPlusWheel = m_plusWheelFpga->getOutputEy();
    m_etValPlusWheel = m_plusWheelFpga->getOutputEt();
    m_htValPlusWheel = m_plusWheelFpga->getOutputHt();
    m_hxValPlusWheel = m_plusWheelJetFpga->getOutputHx();
    m_hyValPlusWheel = m_plusWheelJetFpga->getOutputHy();
  
    m_exVlMinusWheel = m_minusWheelFpga->getOutputEx();
    m_eyVlMinusWheel = m_minusWheelFpga->getOutputEy();
    m_etVlMinusWheel = m_minusWheelFpga->getOutputEt();
    m_htVlMinusWheel = m_minusWheelFpga->getOutputHt();
    m_hxVlMinusWheel = m_minusWheelJetFpga->getOutputHx();
    m_hyVlMinusWheel = m_minusWheelJetFpga->getOutputHy();

    m_hfSumProcessor->fetchInput();
  }
}


// process the event
void L1GctGlobalEnergyAlgos::process()
{
  if (m_setupOk) {
    // Store the inputs in pipelines
    m_exValPlusPipe.store(m_exValPlusWheel, bxRel());
    m_eyValPlusPipe.store(m_eyValPlusWheel, bxRel());
    m_etValPlusPipe.store(m_etValPlusWheel, bxRel());
    m_htValPlusPipe.store(m_htValPlusWheel, bxRel());
    m_hxValPlusPipe.store(m_hxValPlusWheel, bxRel());
    m_hyValPlusPipe.store(m_hyValPlusWheel, bxRel());

    m_exVlMinusPipe.store(m_exVlMinusWheel, bxRel());
    m_eyVlMinusPipe.store(m_eyVlMinusWheel, bxRel());
    m_etVlMinusPipe.store(m_etVlMinusWheel, bxRel());
    m_htVlMinusPipe.store(m_htVlMinusWheel, bxRel());
    m_hxVlMinusPipe.store(m_hxVlMinusWheel, bxRel());
    m_hyVlMinusPipe.store(m_hyVlMinusWheel, bxRel());

    // Process to produce the outputs
    etComponentType ExSum, EySum;
    etmiss_vec EtMissing, HtMissing;

    //
    //-----------------------------------------------------------------------------
    // Form the Ex and Ey sums
    ExSum = m_exValPlusWheel + m_exVlMinusWheel;
    EySum = m_eyValPlusWheel + m_eyVlMinusWheel;
    // Execute the missing Et algorithm
    // Rotate by pi to evaluate MISSING Et. 
    // Implement this in the same way as the firmware
    m_metComponents.setComponents(ExSum, EySum);
    EtMissing = m_metComponents.metVector();
    if (EtMissing.phi.value() > 35) {
      EtMissing.phi.setValue(EtMissing.phi.value() - 36);
    } else {
      EtMissing.phi.setValue(EtMissing.phi.value() + 36);
    }
    if (EtMissing.mag.value() == etMissMaxValue) EtMissing.mag.setOverFlow(true);

    m_outputEtMiss.store    (EtMissing.mag, bxRel());
    m_outputEtMissPhi.store (EtMissing.phi, bxRel());

    //
    //-----------------------------------------------------------------------------
    // Form the Hx and Hy sums
    ExSum = m_hxValPlusWheel + m_hxVlMinusWheel;
    EySum = m_hyValPlusWheel + m_hyVlMinusWheel;
    // Execute the missing Et algorithm
    // Implement this in the same way as the firmware
    m_mhtComponents.setComponents(ExSum, EySum);
    HtMissing = m_mhtComponents.metVector();
    if (HtMissing.phi.value() > 8) {
      HtMissing.phi.setValue(HtMissing.phi.value() - 9);
    } else {
      HtMissing.phi.setValue(HtMissing.phi.value() + 9);
    }

    // Store 7 bits of magnitude and 5 bits of phi angle.
    static const unsigned MAX_HT_VALUE = 0x7f;
    static const unsigned PHI_HT_MASK  = 0x1f;
    if ( (HtMissing.mag.value() > MAX_HT_VALUE) || (HtMissing.mag.overFlow()) ) {
      HtMissing.mag.setValue(MAX_HT_VALUE);
    }
    HtMissing.phi.setValue(HtMissing.phi.value() & PHI_HT_MASK);
    m_outputHtMiss.store    (HtMissing.mag, bxRel());
    m_outputHtMissPhi.store (HtMissing.phi, bxRel());

    //
    //-----------------------------------------------------------------------------
    // Form the Et and Ht sums
    etTotalType ettTemp = m_etValPlusWheel + m_etVlMinusWheel;
    if (ettTemp.overFlow()) ettTemp.setValue(etTotalMaxValue);
    etHadType   httTemp = m_htValPlusWheel + m_htVlMinusWheel;
    if (httTemp.overFlow()) httTemp.setValue(etHadMaxValue);
    m_outputEtSum.store (ettTemp, bxRel());
    m_outputEtHad.store (httTemp, bxRel());

    m_hfSumProcessor->process();
  }
}

//----------------------------------------------------------------------------------------------
// load setup info (for HtMiss calculation)
//
void L1GctGlobalEnergyAlgos::setJetFinderParams(const L1GctJetFinderParams* const jfpars)
{
  // The jetFinders add an LSB when converting to x and y components,
  // so the scale lsb for htx and hty is half the lsb for htt.
  m_mhtComponents.setEtComponentLsb(jfpars->getHtLsbGeV()/2);
}

void L1GctGlobalEnergyAlgos::setHtMissScale(const L1CaloEtScale* const scale)
{
  m_mhtComponents.setEtScale(scale);
}

//----------------------------------------------------------------------------------------------
// check setup
//
bool L1GctGlobalEnergyAlgos::setupOk() const { 
  return (m_setupOk && 
	  m_hfSumProcessor != 0 &&
	  m_hfSumProcessor->setupOk()); 
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
  if (wheel==0) {
    m_htValPlusWheel.setValue(energy);
    m_htValPlusWheel.setOverFlow(overflow);
  } else if (wheel==1) {
    m_htVlMinusWheel.setValue(energy);
    m_htVlMinusWheel.setOverFlow(overflow);
  }
}


//----------------------------------------------------------------------------------------------
// set input data per wheel: x component of Ht
//
void L1GctGlobalEnergyAlgos::setInputWheelHx(unsigned wheel, unsigned energy, bool overflow)
{
  if (wheel==0) {
    m_hxValPlusWheel.setValue(energy);
    m_hxValPlusWheel.setOverFlow(overflow);
  } else if (wheel==1) {
    m_hxVlMinusWheel.setValue(energy);
    m_hxVlMinusWheel.setOverFlow(overflow);
  }
}


//----------------------------------------------------------------------------------------------
// set input data per wheel: y component of Ht
//
void L1GctGlobalEnergyAlgos::setInputWheelHy(unsigned wheel, unsigned energy, bool overflow)
{
  if (wheel==0) {
    m_hyValPlusWheel.setValue(energy);
    m_hyValPlusWheel.setOverFlow(overflow);
  } else if (wheel==1) {
    m_hyVlMinusWheel.setValue(energy);
    m_hyVlMinusWheel.setOverFlow(overflow);
  }
}



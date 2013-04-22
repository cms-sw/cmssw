#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalHfSumAlgos.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfBitCountsLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfEtSumsLut.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

L1GctGlobalHfSumAlgos::L1GctGlobalHfSumAlgos(const std::vector<L1GctWheelJetFpga*>& wheelJetFpga) :
  L1GctProcessor(),
  m_plusWheelJetFpga(wheelJetFpga.at(1)),
  m_minusWheelJetFpga(wheelJetFpga.at(0)),
  m_bitCountLuts(), m_etSumLuts(),
  m_hfInputSumsPlusWheel(),
  m_hfInputSumsMinusWheel(),
  m_hfOutputSumsPipe(),
  m_setupOk(true)
{
  if(wheelJetFpga.size() != 2)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalHfSumAlgos::L1GctGlobalHfSumAlgos() : Global HfSum Algos has been incorrectly constructed!\n"
	  << "This class needs two wheel jet fpga pointers. "
	  << "Number of wheel jet fpga pointers present is " << wheelJetFpga.size() << ".\n";
      }
    }
  
  if(m_plusWheelJetFpga == 0)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalHfSumAlgos::L1GctGlobalHfSumAlgos() has been incorrectly constructed!\n"
	  << "Plus Wheel Jet Fpga pointer has not been set!\n";
      }
    }
  if(m_minusWheelJetFpga == 0)
    {
      m_setupOk = false;
      if (m_verbose) {
	edm::LogWarning("L1GctSetupError")
	  << "L1GctGlobalHfSumAlgos::L1GctGlobalHfSumAlgos() has been incorrectly constructed!\n"
	  << "Minus Wheel Jet Fpga pointer has not been set!\n";
      }
    }

    if (!m_setupOk && m_verbose) {
      edm::LogError("L1GctSetupError") << "L1GctGlobalEnergyAlgos has been incorrectly constructed";
    }
}

L1GctGlobalHfSumAlgos::~L1GctGlobalHfSumAlgos()
{
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfBitCountsLut*>::const_iterator bclut = m_bitCountLuts.begin();
  while (bclut != m_bitCountLuts.end()) {
    delete bclut->second;
    bclut++;
  }
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfEtSumsLut*>::const_iterator eslut = m_etSumLuts.begin();
  while (eslut != m_etSumLuts.end()) {
    delete eslut->second;
    eslut++;
  }
}

std::ostream& operator << (std::ostream& os, const L1GctGlobalHfSumAlgos& fpga)
{
  os << "===L1GctGlobalHfSumAlgos===" << std::endl;
  os << "WheelJetFpga* plus  = " << fpga.m_plusWheelJetFpga << std::endl;
  os << "Plus wheel inputs:" << std::endl;
  os << "Bit counts ring 1: " << fpga.m_hfInputSumsPlusWheel.nOverThreshold0 
     << ", ring 2: " << fpga.m_hfInputSumsPlusWheel.nOverThreshold1 << std::endl;
  os << "Et sums ring 1: " << fpga.m_hfInputSumsPlusWheel.etSum0 
     << ", ring 2: " << fpga.m_hfInputSumsPlusWheel.etSum1 << std::endl;
  os << "WheelJetFpga* minus = " << fpga.m_minusWheelJetFpga << std::endl;
  os << "Minus wheel inputs:" << std::endl;
  os << "Bit counts ring 1: " << fpga.m_hfInputSumsMinusWheel.nOverThreshold0 
     << ", ring 2: " << fpga.m_hfInputSumsMinusWheel.nOverThreshold1 << std::endl;
  os << "Et sums ring 1: " << fpga.m_hfInputSumsMinusWheel.etSum0 
     << ", ring 2: " << fpga.m_hfInputSumsMinusWheel.etSum1 << std::endl;

  int bxZero = -fpga.bxMin();
  if (bxZero>=0 && bxZero<fpga.numOfBx()) {
    os << "Output word " << std::hex << fpga.hfSumsWord().at(bxZero) << std::dec << std::endl;
  }

  return os;
}

void L1GctGlobalHfSumAlgos::resetProcessor() {
  m_hfInputSumsPlusWheel.reset();
  m_hfInputSumsMinusWheel.reset();
}

void L1GctGlobalHfSumAlgos::resetPipelines() {
  m_hfOutputSumsPipe.clear();
  Pipeline<uint16_t> temp(numOfBx());
  // Make one copy of the empty pipeline for each type of Hf lut
  unsigned nTypes = (unsigned) L1GctHfEtSumsLut::numberOfLutTypes;
  for (unsigned t=0; t<nTypes; ++t) {
    m_hfOutputSumsPipe[ (L1GctHfEtSumsLut::hfLutType) t] = temp;
  }
}

void L1GctGlobalHfSumAlgos::fetchInput() {
  if (m_setupOk) {
    m_hfInputSumsPlusWheel  = m_plusWheelJetFpga->getOutputHfSums();
    m_hfInputSumsMinusWheel = m_minusWheelJetFpga->getOutputHfSums();
  }
}


// process the event
void L1GctGlobalHfSumAlgos::process()
{
  if (m_setupOk) {
    // step through the different types of Hf summed quantity
    // and store each one in turn into the relevant pipeline

    // bit count, positive eta, ring 1
    storeBitCount(L1GctHfEtSumsLut::bitCountPosEtaRing1, m_hfInputSumsPlusWheel.nOverThreshold0.value() );

    // bit count, negative eta, ring 1
    storeBitCount(L1GctHfEtSumsLut::bitCountNegEtaRing1, m_hfInputSumsMinusWheel.nOverThreshold0.value() );

    // bit count, positive eta, ring 2
    storeBitCount(L1GctHfEtSumsLut::bitCountPosEtaRing2, m_hfInputSumsPlusWheel.nOverThreshold1.value() );

    // bit count, negative eta, ring 2
    storeBitCount(L1GctHfEtSumsLut::bitCountNegEtaRing2, m_hfInputSumsMinusWheel.nOverThreshold1.value() );

    // et sum, positive eta, ring 1
    storeEtSum(L1GctHfEtSumsLut::etSumPosEtaRing1, m_hfInputSumsPlusWheel.etSum0.value() );

    // et sum, negative eta, ring 1
    storeEtSum(L1GctHfEtSumsLut::etSumNegEtaRing1, m_hfInputSumsMinusWheel.etSum0.value() );

    // et sum, positive eta, ring 2
    storeEtSum(L1GctHfEtSumsLut::etSumPosEtaRing2, m_hfInputSumsPlusWheel.etSum1.value() );

    // et sum, negative eta, ring 2
    storeEtSum(L1GctHfEtSumsLut::etSumNegEtaRing2, m_hfInputSumsMinusWheel.etSum1.value() );

  }
}

// Convert bit count value using LUT and store in the pipeline
void L1GctGlobalHfSumAlgos::storeBitCount(L1GctHfEtSumsLut::hfLutType type, uint16_t value) {
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfBitCountsLut*>::const_iterator bclut = m_bitCountLuts.find(type);
  if (bclut != m_bitCountLuts.end()) {
    m_hfOutputSumsPipe[type].store( (*bclut->second)[value], bxRel() );
  }
}

// Convert et sum value using LUT and store in the pipeline
void L1GctGlobalHfSumAlgos::storeEtSum(L1GctHfEtSumsLut::hfLutType type, uint16_t value) {
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfEtSumsLut*>::const_iterator eslut = m_etSumLuts.find(type);
  if (eslut != m_etSumLuts.end()) {
    m_hfOutputSumsPipe[type].store( (*eslut->second)[value], bxRel() );
  }
}



/// Access to output quantities
std::vector<uint16_t> L1GctGlobalHfSumAlgos::hfSumsOutput(const L1GctHfEtSumsLut::hfLutType type) const
{
  std::vector<uint16_t> result(numOfBx());
  std::map<L1GctHfEtSumsLut::hfLutType, Pipeline<uint16_t> >::const_iterator lut=m_hfOutputSumsPipe.find(type);
  if (lut != m_hfOutputSumsPipe.end()) {
    result = (lut->second).contents;
  }

  return result;

}

std::vector<unsigned> L1GctGlobalHfSumAlgos::hfSumsWord() const {
  std::vector<unsigned> result(numOfBx(), 0x00001000);
  std::vector<uint16_t> outputBits;

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::bitCountPosEtaRing1);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx); }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::bitCountNegEtaRing1);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 3; }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::bitCountPosEtaRing2);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 6; }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::bitCountNegEtaRing2);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 9; }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::etSumPosEtaRing1);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 12; }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::etSumNegEtaRing1);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 16; }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::etSumPosEtaRing2);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 19; }

  outputBits = hfSumsOutput(L1GctHfEtSumsLut::etSumNegEtaRing2);
  for (unsigned bx=0; bx<outputBits.size(); bx++) { result.at(bx) |= outputBits.at(bx) << 22; }

  return result;
}

/// Setup luts
void L1GctGlobalHfSumAlgos::setupLuts(const L1CaloEtScale* scale)
{
  // Replaces existing list of luts with a new one
  while (!m_bitCountLuts.empty()) {
    delete m_bitCountLuts.begin()->second;
    m_bitCountLuts.erase(m_bitCountLuts.begin());
  }
  m_bitCountLuts[L1GctHfEtSumsLut::bitCountPosEtaRing1] = new L1GctHfBitCountsLut(L1GctHfEtSumsLut::bitCountPosEtaRing1);
  m_bitCountLuts[L1GctHfEtSumsLut::bitCountPosEtaRing2] = new L1GctHfBitCountsLut(L1GctHfEtSumsLut::bitCountPosEtaRing2);
  m_bitCountLuts[L1GctHfEtSumsLut::bitCountNegEtaRing1] = new L1GctHfBitCountsLut(L1GctHfEtSumsLut::bitCountNegEtaRing1);
  m_bitCountLuts[L1GctHfEtSumsLut::bitCountNegEtaRing2] = new L1GctHfBitCountsLut(L1GctHfEtSumsLut::bitCountNegEtaRing2);

  while (!m_etSumLuts.empty()) {
    delete m_etSumLuts.begin()->second;
    m_etSumLuts.erase(m_etSumLuts.begin());
  }
  m_etSumLuts[L1GctHfEtSumsLut::etSumPosEtaRing1] = new L1GctHfEtSumsLut(L1GctHfEtSumsLut::etSumPosEtaRing1, scale);
  m_etSumLuts[L1GctHfEtSumsLut::etSumPosEtaRing2] = new L1GctHfEtSumsLut(L1GctHfEtSumsLut::etSumPosEtaRing2, scale);
  m_etSumLuts[L1GctHfEtSumsLut::etSumNegEtaRing1] = new L1GctHfEtSumsLut(L1GctHfEtSumsLut::etSumNegEtaRing1, scale);
  m_etSumLuts[L1GctHfEtSumsLut::etSumNegEtaRing2] = new L1GctHfEtSumsLut(L1GctHfEtSumsLut::etSumNegEtaRing2, scale);

}

/// Get lut pointers
const L1GctHfBitCountsLut* L1GctGlobalHfSumAlgos::getBCLut(const L1GctHfEtSumsLut::hfLutType type) const
{
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfBitCountsLut*>::const_iterator bclut = m_bitCountLuts.find(type);
  if (bclut != m_bitCountLuts.end()) {
    return (bclut->second);
  } else {
   return 0;
  }
}

const L1GctHfEtSumsLut* L1GctGlobalHfSumAlgos::getESLut(const L1GctHfEtSumsLut::hfLutType type) const
{
  std::map<L1GctHfEtSumsLut::hfLutType, const L1GctHfEtSumsLut*>::const_iterator eslut = m_etSumLuts.find(type);
  if (eslut != m_etSumLuts.end()) {
    return (eslut->second);
  } else {
    return 0;
  }
}

/// Get thresholds
std::vector<double> L1GctGlobalHfSumAlgos::getThresholds(const L1GctHfEtSumsLut::hfLutType type) const
{
  std::vector<double> result;
  const L1GctHfEtSumsLut* ESLut = getESLut(type);
  if (ESLut != 0) { result = ESLut->lutFunction()->getThresholds(); }
  return result;
} 


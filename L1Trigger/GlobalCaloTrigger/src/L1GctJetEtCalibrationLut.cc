
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

//DEFINE STATICS
const int L1GctJetEtCalibrationLut::NAddress=JET_ET_CAL_LUT_ADD_BITS;
const int L1GctJetEtCalibrationLut::NData=JET_ET_CAL_LUT_DAT_BITS;
const unsigned L1GctJetEtCalibrationLut::JET_ENERGY_BITWIDTH = 10;

L1GctJetEtCalibrationLut::L1GctJetEtCalibrationLut() :
  L1GctLut<NAddress,NData>()
{
}


L1GctJetEtCalibrationLut::~L1GctJetEtCalibrationLut()
{
}

void L1GctJetEtCalibrationLut::setFunction(const L1GctJetFinderParams* const lutfn)
{
  m_lutFunction = lutfn;
  m_setupOk = (lutfn!=0);
}

void L1GctJetEtCalibrationLut::setOutputEtScale(const L1CaloEtScale* const scale) {
  m_outputEtScale = scale;
}

void L1GctJetEtCalibrationLut::setEtaBin(const unsigned eta) {
  static const unsigned nEtaBits = 4;
  static const uint8_t etaMask    = static_cast<uint8_t>((1 << nEtaBits) - 1);
  m_etaBin = static_cast<uint8_t>(eta) & etaMask;
}

uint16_t L1GctJetEtCalibrationLut::value (const uint16_t lutAddress) const
{
  static const uint16_t maxEtMask  = static_cast<uint16_t>((1 << JET_ENERGY_BITWIDTH) - 1);
  static const uint16_t tauBitMask = static_cast<uint16_t>( 1 << (JET_ENERGY_BITWIDTH));
  static const uint16_t ovrFlowOut = 0x3f;
  uint16_t jetEt = lutAddress & maxEtMask;
  // Check for saturation
  if (jetEt == maxEtMask) {
    return ovrFlowOut;
  } else {
    double uncoEt = static_cast<double>(jetEt) * m_outputEtScale->linearLsb();
    bool tauVeto = ((lutAddress & tauBitMask)==0);
  
    double corrEt = m_lutFunction->correctedEtGeV(uncoEt, etaBin(), tauVeto);
    return m_outputEtScale->rank(corrEt);
  }
}

std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationLut& lut)
{
  os << std::endl;
  os << "==================================================" << std::endl;
  os << "===Level-1 Trigger:  GCT Jet Et Calibration Lut===" << std::endl;
  os << "==================================================" << std::endl;
  os << "===Parameter settings for eta bin " << lut.etaBin() << "===" << std::endl;
  os << *lut.getFunction() << std::endl;
  os << "\n===Lookup table contents===\n" << std::endl;
  const L1GctLut<L1GctJetEtCalibrationLut::NAddress,L1GctJetEtCalibrationLut::NData>* temp=&lut;
  os << *temp;
  return os;
}

template class L1GctLut<L1GctJetEtCalibrationLut::NAddress,L1GctJetEtCalibrationLut::NData>;


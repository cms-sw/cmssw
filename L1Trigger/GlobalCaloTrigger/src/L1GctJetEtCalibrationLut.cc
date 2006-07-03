#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

//DEFINE STATICS
const unsigned L1GctJetEtCalibrationLut::JET_ENERGY_BITWIDTH = L1GctJet::RAWSUM_BITWIDTH;

L1GctJetEtCalibrationLut::L1GctJetEtCalibrationLut()
{
}

L1GctJetEtCalibrationLut::~L1GctJetEtCalibrationLut()
{
}

uint16_t L1GctJetEtCalibrationLut::convertToSixBitRank(uint16_t jetEnergy, unsigned eta) const
{
  if(jetEnergy < (1 << JET_ENERGY_BITWIDTH))
  {
    return jetEnergy/16;
  }
  return 63;
}

uint16_t L1GctJetEtCalibrationLut::convertToTenBitRank(uint16_t jetEnergy, unsigned eta) const
{
  if(jetEnergy < (1 << JET_ENERGY_BITWIDTH))
  {
    return jetEnergy;
  }
  return 1023;
}

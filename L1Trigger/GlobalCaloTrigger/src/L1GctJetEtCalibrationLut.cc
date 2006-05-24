#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

L1GctJetEtCalibrationLut::L1GctJetEtCalibrationLut()
{
}

L1GctJetEtCalibrationLut::~L1GctJetEtCalibrationLut()
{
}

uint16_t L1GctJetEtCalibrationLut::convertToSixBitRank(uint16_t jetEnergy, uint16_t eta)
{
  if(jetEnergy < (1 << JET_ENERGY_BITWIDTH))
  {
    return jetEnergy/16
  }
  return 63;
}

uint16_t L1GctJetEtCalibrationLut::convertToTenBitRank(uint16_t jetEnergy, uint16_t eta)
{
  if(jetEnergy < (1 << JET_ENERGY_BITWIDTH))
  {
    return jetEnergy;
  }
  return 1023;
}

#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>

EcalDCCHeaderBlock::EcalDCCHeaderBlock()
{
}

EcalDCCHeaderBlock::EcalDCCHeaderBlock(const int& dccId)
{
  dccId_=dccId;
  tccStatus_.reserve(MAX_TCC_SIZE);
  triggerTowerStatus_.reserve(MAX_TT_SIZE);
  dccErrors_=0;
  orbitNumber_=0; // do we need it here?
  cycleSettings_=0;
  runType_=0;
  sequence_=0;
  rtHalf_=0;
  mgpaGain_=0;
  memGain_=0;

  selectiveReadout_=false;
  testZeroSuppression_=false;
  zeroSuppression_=false;  
}



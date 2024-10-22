#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>

EcalDCCHeaderBlock::EcalDCCHeaderBlock() {
  dccId_ = -1;            // initialize
  fedId_ = -1;            // initialize
  dccInTTCCommand_ = -1;  // initialize
  tccStatus_.reserve(MAX_TCC_SIZE);
  triggerTowerFlag_.reserve(MAX_TT_SIZE);
  feStatus_.reserve(MAX_TT_SIZE);

  feBx_.reserve(MAX_TT_SIZE);
  feLv1_.reserve(MAX_TT_SIZE);
  tccBx_.reserve(MAX_TCC_SIZE);
  tccLv1_.reserve(MAX_TCC_SIZE);

  srpLv1_ = -1;
  srpBx_ = -1;

  dccErrors_ = -1;
  orbitNumber_ = -1;  // do we need it here?
  runType_ = -1;
  zs_ = -1;
  basic_trigger_type_ = -1;
  LV1event_ = -1;
  runNumber_ = -1;
  BX_ = -1;

  EcalDCCEventSettings dummySettings;
  dummySettings.LaserPower = -1;
  dummySettings.LaserFilter = -1;
  dummySettings.wavelength = -1;
  dummySettings.delay = -1;
  dummySettings.MEMVinj = -1;
  dummySettings.mgpa_content = -1;
  dummySettings.ped_offset = -1;

  EventSettings_ = dummySettings;

  rtHalf_ = -1;
  mgpaGain_ = -1;
  memGain_ = -1;
  srpStatus_ = -1;
  selectiveReadout_ = false;
  testZeroSuppression_ = false;
  zeroSuppression_ = false;
}

EcalDCCHeaderBlock::EcalDCCHeaderBlock(const int& dccId) {
  dccId_ = dccId;
  fedId_ = -1;  // initialize
  tccStatus_.reserve(MAX_TCC_SIZE);
  triggerTowerFlag_.reserve(MAX_TT_SIZE);
  feStatus_.reserve(MAX_TT_SIZE);

  feBx_.reserve(MAX_TT_SIZE);
  feLv1_.reserve(MAX_TT_SIZE);
  tccBx_.reserve(MAX_TCC_SIZE);
  tccLv1_.reserve(MAX_TCC_SIZE);

  srpLv1_ = -1;
  srpBx_ = -1;

  dccErrors_ = -1;
  orbitNumber_ = -1;  // do we need it here?
  runType_ = -1;
  basic_trigger_type_ = -1;
  LV1event_ = -1;
  runNumber_ = -1;
  BX_ = -1;

  EcalDCCEventSettings dummySettings;
  dummySettings.LaserPower = -1;
  dummySettings.LaserFilter = -1;
  dummySettings.wavelength = -1;
  dummySettings.delay = -1;
  dummySettings.MEMVinj = -1;
  dummySettings.mgpa_content = -1;
  dummySettings.ped_offset = -1;

  EventSettings_ = dummySettings;

  rtHalf_ = -1;
  mgpaGain_ = -1;
  memGain_ = -1;
  srpStatus_ = -1;

  selectiveReadout_ = false;
  testZeroSuppression_ = false;
  zeroSuppression_ = false;
}

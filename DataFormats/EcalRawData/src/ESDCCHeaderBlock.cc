#include "DataFormats/EcalRawData/interface/ESDCCHeaderBlock.h"

ESDCCHeaderBlock::ESDCCHeaderBlock()
{

  dccId_ = -1;
  fedId_ = -1;
  LV1_ = -1;
  BX_ = -1;
  gain_ = -1;
  precision_ = -1;
  dac_ = -1;
  evtLen_ = -1;
  dccErrs_ = -1;
  runNum_ = -1;
  runType_ = -1;
  seqType_ = -1;
  trgType_ = -1;
  compFlag_ = -1;
  orbit_ = -1;
  vmajor_ = -1;
  vminor_ = -1;
  optoRX0_ = -1;
  optoRX1_ = -1;
  optoRX2_ = -1;
  optoBC0_ = -1;
  optoBC1_ = -1;
  optoBC2_ = -1;
  FEch_.reserve(36);
  packetLen_ = -1;
  bc_ = -1;
  ev_ = -1;
  BMMeasurements_ = -1;
  beginOfSpillSec_ = -1;
  beginOfSpillMilliSec_ = -1;
  endOfSpillSec_ = -1;
  endOfSpillMilliSec_ = -1;
  beginOfSpillLV1_ = -1;
  endOfSpillLV1_ = -1;
  timestamp_sec_ = -1;
  timestamp_usec_ = -1;
  spillNum_ = -1;
  evtInSpill_ = -1;
  camacErr_ = -1;
  vmeErr_ = -1;
  ADCch_status_.reserve(12);
  ADCch_.reserve(12);
  TDCch_status_.reserve(8);
  TDCch_.reserve(8);

}

ESDCCHeaderBlock::ESDCCHeaderBlock(const int& dccId)
{

  dccId_ = dccId;
  fedId_ = -1;
  LV1_ = -1;
  BX_ = -1;
  gain_ = -1;
  precision_ = -1;
  dac_ = -1;
  evtLen_ = -1;
  dccErrs_ = -1;
  runNum_ = -1;
  runType_ = -1;
  seqType_ = -1;
  trgType_ = -1;
  compFlag_ = -1;
  orbit_ = -1;
  vmajor_ = -1;
  vminor_ = -1;
  optoRX0_ = -1;
  optoRX1_ = -1;
  optoRX2_ = -1;
  optoBC0_ = -1;
  optoBC1_ = -1;
  optoBC2_ = -1;
  FEch_.reserve(36);
  packetLen_ = -1;
  bc_ = -1;
  ev_ = -1;
  BMMeasurements_ = -1;
  beginOfSpillSec_ = -1;
  beginOfSpillMilliSec_ = -1;
  endOfSpillSec_ = -1;
  endOfSpillMilliSec_ = -1;
  beginOfSpillLV1_ = -1;
  endOfSpillLV1_ = -1;
  timestamp_sec_ = -1;
  timestamp_usec_ = -1;
  spillNum_ = -1;
  evtInSpill_ = -1;
  camacErr_ = -1;
  vmeErr_ = -1;
  ADCch_status_.reserve(12);
  ADCch_.reserve(12);
  TDCch_status_.reserve(8);
  TDCch_.reserve(8);

}

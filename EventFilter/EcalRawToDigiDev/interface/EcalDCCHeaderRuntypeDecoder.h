// Date   : 13/06/2007

#ifndef ECALDCCHEADERRUNTYPE_DECODER_H
#define ECALDCCHEADERRUNTYPE_DECODER_H
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>
#include "DCCRawDataDefinitions.h"

class EcalDCCHeaderRuntypeDecoder
{
 public:
  EcalDCCHeaderRuntypeDecoder();
  ~EcalDCCHeaderRuntypeDecoder();
  bool Decode( ulong TrTy, ulong detTrTy, ulong runType,   EcalDCCHeaderBlock * theHeader);
  protected:
  bool WasDecodingOk_;
  void DecodeSetting ( int settings,  EcalDCCHeaderBlock * theHeader );
  void DecodeSettingGlobal ( ulong TrigType, ulong detTrigType,  EcalDCCHeaderBlock * theHeader );
  void CleanEcalDCCSettingsInfo(  EcalDCCHeaderBlock::EcalDCCEventSettings * theEventSettings);// Re-initialize theEventSettings  before filling with the deocoded event

};

#endif

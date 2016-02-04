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
  bool Decode( unsigned long TrTy, unsigned long detTrTy, unsigned long runType,   EcalDCCHeaderBlock * theHeader);
  protected:
  bool WasDecodingOk_;
  void DecodeSetting ( int settings,  EcalDCCHeaderBlock * theHeader );
  void DecodeSettingGlobal ( unsigned long TrigType, unsigned long detTrigType,  EcalDCCHeaderBlock * theHeader );
  void CleanEcalDCCSettingsInfo(  EcalDCCHeaderBlock::EcalDCCEventSettings * theEventSettings);// Re-initialize theEventSettings  before filling with the deocoded event

};

#endif

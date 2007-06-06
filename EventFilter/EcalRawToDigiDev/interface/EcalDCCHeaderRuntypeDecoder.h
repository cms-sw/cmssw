// Date   : 13/06/2005

#ifndef ECALDCCHEADERRUNTYPE_DECODER_H
#define ECALDCCHEADERRUNTYPE_DECODER_H
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>
class EcalDCCHeaderRuntypeDecoder
{
 public:
  EcalDCCHeaderRuntypeDecoder();
  ~EcalDCCHeaderRuntypeDecoder();
  bool Decode( ulong headerWord,   EcalDCCHeaderBlock * theHeader);
  protected:
  bool WasDecodingOk_;
  void DecodeSetting ( int settings,  EcalDCCHeaderBlock * theHeader );
  void CleanEcalDCCSettingsInfo(  EcalDCCHeaderBlock::EcalDCCEventSettings * theEventSettings);// Re-initialize theEventSettings  before filling with the deocoded event
};
#endif

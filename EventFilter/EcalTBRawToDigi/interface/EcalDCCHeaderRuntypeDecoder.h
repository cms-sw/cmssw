// Date   : 13/06/2005

#ifndef ECALDCCTBHEADERRUNTYPE_DECODER_H
#define ECALDCCTBHEADERRUNTYPE_DECODER_H
#include <DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h>
class EcalDCCTBHeaderRuntypeDecoder
{
 public:
  EcalDCCTBHeaderRuntypeDecoder();
  ~EcalDCCTBHeaderRuntypeDecoder();
  bool Decode( unsigned long headerWord,   EcalDCCHeaderBlock * theHeader);
  protected:
  bool WasDecodingOk_;
  void DecodeSetting ( int settings,  EcalDCCHeaderBlock * theHeader );
  void CleanEcalDCCSettingsInfo(  EcalDCCHeaderBlock::EcalDCCEventSettings * theEventSettings);// Re-initialize theEventSettings  before filling with the deocoded event
};
#endif

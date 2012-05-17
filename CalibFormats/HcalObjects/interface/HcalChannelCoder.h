#ifndef HCAL_CHANNEL_CODER_H
#define HCAL_CHANNEL_CODER_H

/** \class HcalChannelCoder
    
    Container for ADC<->fQ conversion constants for HCAL QIE
   $Author: ratnikov
   $Date: 2011/01/21 22:24:37 $
   $Revision: 1.3.6.1 $
*/

class QieShape;

class HcalChannelCoder {
 public:
  HcalChannelCoder (const float fOffset [16], const float fSlope [16]); // [CapId][Range]
  /// ADC[0..255]+capid[0..3]->fC conversion
  double charge (const QieShape& fShape, int fAdc, int fCapId) const;
  /// fC + capid[0..3] -> ADC conversion
  int adc (const QieShape& fShape, double fCharge, int fCapId) const;
  int index (int fCapId, int Range) {return fCapId*4+Range;}
 private:
  double mOffset [4][4];
  double mSlope [4][4];
};

#endif

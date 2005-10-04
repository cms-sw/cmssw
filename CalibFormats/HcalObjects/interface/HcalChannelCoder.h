#ifndef HCAL_CHANNEL_CODER_H
#define HCAL_CHANNEL_CODER_H

/** \class HcalChannelCoder
    
    Container for ADC<->fQ conversion constants for HCAL QIE
   $Author: ratnikov
   $Date: 2005/08/18 23:41:41 $
   $Revision: 1.1 $
*/

class QieShape;

class HcalChannelCoder {
 public:
  HcalChannelCoder (const float fOffset [16], const float fSlope [16]); // [CapId][Range]
  double charge (const QieShape& fShape, int fAdc, int fCapId) const;
  int adc (const QieShape& fShape, double fCharge, int fCapId) const;
  int index (int fCapId, int Range) {return fCapId*4+Range;}
 private:
  double mOffset [4][4];
  double mSlope [4][4];
};

#endif

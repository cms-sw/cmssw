#ifndef HCAL_CHANNEL_CODER_H
#define HCAL_CHANNEL_CODER_H

/** \class HcalChannelCoder
    
    Container for ADC<->fQ conversion constants for HCAL QIE
   $Author: ratnikov
   $Date: 2005/08/02 01:31:24 $
   $Revision: 1.2 $
*/

class QieShape;

class HcalChannelCoder {
 public:
  HcalChannelCoder (double fOffset [4][4], double fSlope [4][4]); // [CapId][Range]
  double charge (const QieShape& fShape, int fAdc, int fCapId) const;
  int adc (const QieShape& fShape, double fCharge, int fCapId) const;
 private:
  double mOffset [4][4];
  double mSlope [4][4];
};

#endif

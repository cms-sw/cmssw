/** \class QieShape
    
    basic linearization function for HCAL QIE
   $Author: ratnikov
   $Date: 2011/01/21 22:24:38 $
   $Revision: 1.2.6.1 $
*/

#include <iostream>

#include "CalibFormats/HcalObjects/interface/QieShape.h"

QieShape::QieShape (const double fAdcShape [64], const double fAdcBin [64]) {
  for (int i = 0; i < 64; i++) {  // initial settings
    mLinearization [i] = fAdcShape [i];
    mBinSize [i] = fAdcBin [i];
    //    std::cout << "QieShape::QieShape-> #/adc/bin: " << i << '/' << fAdcShape [i] << '/' << fAdcBin [i] << std::endl;
  }
  double factor = 1;
  for (int range = 1; range < 4; range++) {
    factor = factor * 8;
    int offset = 64 * range;
    mLinearization [offset] = mLinearization[offset-2]; // initial overlap
    for (int bin = 1; bin < 64; bin++) {
      mLinearization [offset+bin] = mLinearization [offset+bin-1] +
        factor * (mLinearization [bin] - mLinearization [bin-1]); // scale initial curve
      mBinSize [offset+bin] = factor * mBinSize [bin];
    }
  }
}

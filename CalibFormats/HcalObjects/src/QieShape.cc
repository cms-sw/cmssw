/** \class QieShape
    
    basic linearization function for HCAL QIE
   $Author: ratnikov
   $Date: 2005/11/07 22:15:09 $
   $Revision: 1.2 $
*/

#include <iostream>

#include "CalibFormats/HcalObjects/interface/QieShape.h"

QieShape::QieShape (const double fAdcShape [32], const double fAdcBin [32]) {
  for (int i = 0; i < 32; i++) {  // initial settings
    mLinearization [i] = fAdcShape [i];
    mBinSize [i] = fAdcBin [i];
    //    std::cout << "QieShape::QieShape-> #/adc/bin: " << i << '/' << fAdcShape [i] << '/' << fAdcBin [i] << std::endl;
  }
  double factor = 1;
  for (int range = 1; range < 4; range++) {
    factor = factor * 5;
    int offset = 32 * range;
    mLinearization [offset] = mLinearization[offset-2]; // initial overlap
    for (int bin = 1; bin < 32; bin++) {
      mLinearization [offset+bin] = mLinearization [offset+bin-1] +
        factor * (mLinearization [bin] - mLinearization [bin-1]); // scale initial curve
      mBinSize [offset+bin] = factor * mBinSize [bin];
    }
  }
}

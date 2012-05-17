#ifndef QIE_SHAPE_H
#define QIE_SHAPE_H

/** \class QieShape
    
    basic linearization function for HCAL QIE
   $Author: ratnikov
   $Date: 2011/01/21 22:24:37 $
   $Revision: 1.1.6.1 $
*/
class QieShape {
 public:
  QieShape (const double fAdcShape [64], const double fAdcBin [64]);
  // center of the nominal linearized QIE bin
  double linearization (int fAdc) const {return mLinearization [fAdc];}
  // width of the nominal linearized bin
  double binSize (int fAdc) const {return mBinSize [fAdc];}
 private:
  double mLinearization [256];
  double mBinSize [256];
};

#endif
